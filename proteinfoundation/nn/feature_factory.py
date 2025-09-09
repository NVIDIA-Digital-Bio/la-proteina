import gzip
import math
import os
import random
from typing import Dict, List, Literal

import einops
import torch
from jaxtyping import Float
from loguru import logger
from torch.nn import functional as F
from torch_scatter import scatter_mean

from openfold.data import data_transforms
from openfold.np.residue_constants import atom_types
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle
from proteinfoundation.utils.fold_utils import extract_cath_code_by_level
from torch.nn.utils.rnn import pad_sequence


################################
# # Some auxiliary functions # #
################################


# From frameflow code
def get_index_embedding(indices, edim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of type integer, shape either [n] or [b, n].
        edim: dimension of the embeddings to create.
        max_len: maximum length.

    Returns:
        positional embedding of shape either [n, edim] or [b, n, edim]
    """
    # indices [n] of [b, n]
    K = torch.arange(edim // 2, device=indices.device)  # [edim / 2]

    if len(indices.shape) == 1:  # [n]
        K = K[None, ...]
    elif len(indices.shape) == 2:  # [b, n]
        K = K[None, None, ...]

    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K / edim))
    ).to(indices.device)
    # [n, 1] / [1, edim/2] -> [n, edim/2] or [b, n, 1] / [1, 1, edim/2] -> [b, n, edim/2]
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K / edim))
    ).to(indices.device)
    pos_embedding = torch.cat(
        [pos_embedding_sin, pos_embedding_cos], axis=-1
    )  # [n, edim]
    return pos_embedding


def get_time_embedding(
    t: Float[torch.Tensor, "b"], edim: int, max_positions: int = 2000
) -> Float[torch.Tensor, "b embdim"]:
    """
    Code from Frameflow, which got it from
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: Dict):
        pass  # Implemented by each class

    def extract_bs_and_n(self, batch: Dict):
        """
        Extracts batch size and n from input batch.
        Once we add more modalities just try modaliites until you hit one.
        Other option is to pass arguments to nn which modalities are used.
        """
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise IOError("Don't know how to extract batch size and n from batch...")
        bs, n = v.shape[0], v.shape[1]
        return bs, n

    def extract_device(self, batch: Dict):
        """Extracts device from input batch."""
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise IOError("Don't know how to extract device from batch...")
        return v.device

    def assert_defaults_allowed(self, batch: Dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise IOError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(
        self,
        dim_feats_out=128,
        mode: Literal["seq", "pair"] = "seq",
        name=None,
        **kwargs,
    ):
        super().__init__(dim=128)
        self.mode = mode

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=device)
        elif self.mode == "pair":
            torch.zeros((b, n, n, self.dim_feats_out), device=device)
        else:
            raise IOError(f"Mode {self.mode} wrong for zero feature")


class CroppedFlagSeqFeat(Feature):
    """Computes feature of shape [b, n, 1] indicating if protein is cropped
    (1s if it is, 0s if it isn't)."""

    def __init__(self):
        super().__init__(dim=1)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if "cropped" in batch:
            ones = torch.ones((b, n, self.dim), device=device)
            cropped = batch["cropped"]  # boolean [b]
            return ones * cropped[..., None, None]  # [b, n, dim(=1)]
        else:
            return torch.zeros((b, n, self.dim), device=device)


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding and returns as sequence feature of shape [b, n, fold_emb_dim * 3]."""

    def __init__(
        self,
        fold_emb_dim,
        cath_code_dir,
        multilabel_mode="sample",
        fold_nhead=4,
        fold_nlayer=2,
        **kwargs,
    ):
        """
        multilabel_mode (["sample", "average", "transformer"]): Schemes to handle multiple fold labels
            "sample": randomly sample one label
            "average": average fold embeddings over all labels
            "transformer": pad labels together and feed into a transformer, take the average over the output
        """
        super().__init__(dim=fold_emb_dim * 3)
        self.create_mapping(cath_code_dir)
        self.embedding_C = torch.nn.Embedding(
            self.num_classes_C + 1, fold_emb_dim
        )  # The last class is left as null embedding
        self.embedding_fA = torch.nn.Embedding(self.num_classes_A + 1, fold_emb_dim)
        self.embedding_T = torch.nn.Embedding(self.num_classes_T + 1, fold_emb_dim)
        self.register_buffer("_device_param", torch.tensor(0), persistent=False)
        assert multilabel_mode in ["sample", "average", "transformer"]
        self.multilabel_mode = multilabel_mode
        if multilabel_mode == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                fold_emb_dim * 3,
                nhead=fold_nhead,
                dim_feedforward=fold_emb_dim * 3,
                batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, fold_nlayer)

    @property
    def device(self):
        return next(self.buffers()).device

    def create_mapping(self, cath_code_dir):
        """Create cath label vocabulary for C, A, T levels."""
        mapping_file = os.path.join(cath_code_dir, "cath_label_mapping.pt")
        if os.path.exists(mapping_file):
            class_mapping = torch.load(mapping_file)
        else:
            cath_code_file = os.path.join(cath_code_dir, "cath-b-newest-all.gz")
            cath_code_set = {"C": set(), "A": set(), "T": set()}
            with gzip.open(cath_code_file, "rt") as f:
                for line in f:
                    cath_id, cath_version, cath_code, cath_segment_and_chain = (
                        line.strip().split()
                    )
                    cath_code_set["C"].add(extract_cath_code_by_level(cath_code, "C"))
                    cath_code_set["A"].add(extract_cath_code_by_level(cath_code, "A"))
                    cath_code_set["T"].add(extract_cath_code_by_level(cath_code, "T"))
            class_mapping = {
                "C": {k: i for i, k in enumerate(sorted(list(cath_code_set["C"])))},
                "A": {k: i for i, k in enumerate(sorted(list(cath_code_set["A"])))},
                "T": {k: i for i, k in enumerate(sorted(list(cath_code_set["T"])))},
            }
            torch.save(class_mapping, mapping_file)

        self.class_mapping_C = class_mapping["C"]
        self.class_mapping_A = class_mapping["A"]
        self.class_mapping_T = class_mapping["T"]
        self.num_classes_C = len(self.class_mapping_C)
        self.num_classes_A = len(self.class_mapping_A)
        self.num_classes_T = len(self.class_mapping_T)

    def parse_label(self, cath_code_list):
        """Parse cath_code into corresponding indices at C, A, T levels

        Args:
            cath_code_list (List[List[str]]): List of cath codes for each protein. Each protein can have no, one or multiple labels.

        Return:
            results: for each label of each protein, return its C, A, T label indices
        """
        results = []
        for cath_codes in cath_code_list:
            result = []
            for cath_code in cath_codes:
                result.append(
                    [
                        self.class_mapping_C.get(
                            extract_cath_code_by_level(cath_code, "C"),
                            self.num_classes_C,
                        ),  # If unknown or masked, set as null
                        self.class_mapping_A.get(
                            extract_cath_code_by_level(cath_code, "A"),
                            self.num_classes_A,
                        ),
                        self.class_mapping_T.get(
                            extract_cath_code_by_level(cath_code, "T"),
                            self.num_classes_T,
                        ),
                    ]
                )
            if len(cath_codes) == 0:
                result = [
                    [self.num_classes_C, self.num_classes_A, self.num_classes_T]
                ]  # If no cath code is provided, return null
            results.append(result)
        return results  # [b, num_label, 3]

    def sample(self, cath_code_list):
        """Randomly sample one cath code"""
        results = []
        for cath_codes in cath_code_list:
            idx = random.randint(0, len(cath_codes) - 1)
            results.append(cath_codes[idx])
        return results

    def flatten(self, cath_code_list):
        """Flatten variable lengths of cath codes into a long cath code tensor"""
        results = []
        batch_id = []
        for i, cath_codes in enumerate(cath_code_list):
            results += cath_codes
            batch_id += [i] * len(cath_codes)
        results = torch.as_tensor(results, device=self.device)
        batch_id = torch.as_tensor(batch_id, device=self.device)
        return results, batch_id

    def pad(self, cath_code_list):
        """Pad variable lengths of cath codes into a batched cath code tensor"""
        results = []
        max_num_label = 0
        for cath_codes in cath_code_list:
            results.append(cath_codes)
            max_num_label = max(max_num_label, len(cath_codes))
        mask = []
        for i in range(len(results)):
            mask_i = [False] * len(results[i])
            if len(results[i]) < max_num_label:
                mask_i += [True] * (max_num_label - len(results[i]))
                results[i] += [
                    [self.num_classes_C, self.num_classes_A, self.num_classes_T]
                ] * (max_num_label - len(results[i]))
            mask.append(mask_i)
        results = torch.as_tensor(results, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        return results, mask

    def forward(self, batch):
        bs, n = self.extract_bs_and_n(batch)
        if "cath_code" not in batch:
            cath_code = [
                ["x.x.x.x"]
            ] * bs  # If no cath code provided, return null embeddings
        else:
            cath_code = batch["cath_code"]

        cath_code_list = self.parse_label(cath_code)
        if self.multilabel_mode == "sample":
            cath_code_list = self.sample(
                cath_code_list
            )  # Random sample one label for each protein
            cath_code = torch.as_tensor(cath_code_list, device=self.device)  # [b, 3]
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, 0]),
                    self.embedding_A(cath_code[:, 1]),
                    self.embedding_T(cath_code[:, 2]),
                ],
                dim=-1,
            )  # [b, fold_emb_dim * 3]
        elif self.multilabel_mode == "average":
            cath_code, batch_id = self.flatten(cath_code_list)
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, 0]),
                    self.embedding_A(cath_code[:, 1]),
                    self.embedding_T(cath_code[:, 2]),
                ],
                dim=-1,
            )  # [num_code, fold_emb_dim * 3]
            fold_emb = scatter_mean(fold_emb, batch_id, dim=0, dim_size=bs)
        elif self.multilabel_mode == "transformer":
            cath_code, mask = self.pad(cath_code_list)
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, :, 0]),
                    self.embedding_A(cath_code[:, :, 1]),
                    self.embedding_T(cath_code[:, :, 2]),
                ],
                dim=-1,
            )  # [b, max_num_label, fold_emb_dim * 3]
            fold_emb = self.transformer(
                fold_emb, src_key_padding_mask=mask
            )  # [b, max_num_label, fold_emb_dim * 3]
            fold_emb = (fold_emb * (~mask[:, :, None]).float()).sum(dim=1) / (
                (~mask[:, :, None]).float().sum(dim=1) + 1e-10
            )  # [b, fold_emb_dim * 3]
        fold_emb = fold_emb[:, None, :]  # [b, 1, fold_emb_dim * 3]
        return fold_emb.expand(
            (fold_emb.shape[0], n, fold_emb.shape[2])
        )  # [b, n, fold_emb_dim * 3]


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            chain_breaks = torch.zeros((b, n), device=device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscBBCASeqFeat(Feature):
    """Computes feature from backbone CA self conditining coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, mode_key="x_sc", **kwargs):
        super().__init__(dim=3)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "bb_ca" in data_modes_avail
            ), f"`bb_ca` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            return batch[self.mode_key]["bb_ca"]  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {self.mode_key} in batch, returning zeros for XscBBCASeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, 3, device=device)


class XscLocalLatentsSeqFeat(Feature):
    """Computes feature from local latents self conditining, seq feature of shape [b, n, dim]."""

    def __init__(self, latent_dim, mode_key="x_sc", **kwargs):
        super().__init__(dim=latent_dim)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "local_latents" in data_modes_avail
            ), f"`local_latents` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            return batch[self.mode_key]["local_latents"]  # [b, n, latent_dim]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {self.mode_key} in batch, returning zeros for XscLocalLatentsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class XtBBCASeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["bb_ca"]  # [b, n, 3]


class XtLocalLatentsSeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, latent_dim, **kwargs):
        super().__init__(dim=latent_dim)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "local_latents" in data_modes_avail
        ), f"`local_latents` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["local_latents"]  # [b, n, latent_dim]


class CaCoorsNanometersSeqFeat(Feature):
    """Computes feature from ca coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), "`ca_coors_nm` nor `coords_nm` in batch, cannot compute CaCoorsNanometersSeqFeat"
        if "ca_coors_nm" in batch:
            return batch["ca_coors_nm"]  # [b, n, 3]
        else:
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]


class TryCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `ca_coors_nm` in batch, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `ca_coors_nm` not in batch return zero feature.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if "ca_coors_nm" in batch or "coords_nm" in batch:
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No ca_coors_nm or coords_nm in batch, returning zeros for TryCaCoorsNanometersSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class OptionalCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `use_ca_coors_nm_feature` in batch and true, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `use_ca_coors_nm_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if batch.get("use_ca_coors_nm_feature", False):  # defaults to False
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "use_ca_coors_nm_feature disabled or not in batch, returning zeros for OptionalCaCoorsNanometersSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class ResidueTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 20].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=20)

    def forward(self, batch):
        assert (
            "residue_type" in batch
        ), "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        rtype = rtype * rpadmask  # [b, n], the -1 padding becomes 0
        rtype_onehot = F.one_hot(rtype, num_classes=20)  # [b, n, 20]
        rtype_onehot = (
            rtype_onehot * rpadmask[..., None]
        )
        return rtype_onehot * 1.0


class OptionalResidueTypeSeqFeat(ResidueTypeSeqFeat):
    """
    If `use_residue_type_feature` in batch and true, adds residue type feature of shape [b, n, 20].

    If `use_residue_type_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if batch.get("use_residue_type_feature", False):
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "use_residue_type_feature disabled or not in batch, returning zeros for OptionalResidueTypeSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, 20, device=device)


class Atom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from the atom37 representation (in Å), feature of shape [b, n, 37 * 4].

    Atom37 has shape [b, n, 37, 3], and the appropriate mask (for the residue type) has shape
    [b, n, 37]. This feature concatenates the flattened mask (shape [b, n, 37]) with the flattened coordinates (of shape
    [b, n, 37 * 3])

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, rel=False, **kwargs):
        super().__init__(dim=int(37 * 4))
        # 37 * 4, 37 * 3 for the coordinates, + 37 for the mask
        self.rel = rel  # whether to get features relative to CA or absolute

    def forward(self, batch):
        assert (
            "coords_nm" in batch
        ), "`coords_nm` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        assert (
            "coord_mask" in batch
        ), "`coord_mask` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 37, 3]
        coors_mask = batch[
            "coord_mask"
        ]  # [b, n, 37]
        coors = (
            coors * coors_mask[..., None]
        )  # Zero-out non-atoms

        if self.rel:
            # If relative remove CA coordinates
            ca_coors = coors[:, :, 1, :]  # [b, n, 3]
            coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
            coors = coors * coors_mask[..., None]

        coors_flat = einops.rearrange(
            coors, "b n a t -> b n (a t)"
        )  # [b, n, 37, 3] -> [b, n, 37 * 3]
        feat = torch.cat([coors_flat, coors_mask], dim=-1)  # [b, n, 37 * 4]
        return feat


class BackboneTorsionAnglesSeqFeat(Feature):
    """
    Computes torsion angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(3 * 21))

    def forward(self, batch):
        bb_torsion = self._get_bb_torsion_angles(batch)  # [b, n, 3]
        bb_torsion_feats = bin_and_one_hot(
            bb_torsion,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_torsion.device),
        )  # [b, n, 3, nbins], nbins in 20+1
        bb_torsion_feats = einops.rearrange(
            bb_torsion_feats, "b n t d -> b n (t d)"
        )  # [b, n, 3 * nbins]
        return bb_torsion_feats

    def _get_bb_torsion_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]
        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]

        psi = signed_dihedral_angle(
            N[:, :-1, :], CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :]
        )  # [b, n-1]
        omega = signed_dihedral_angle(
            CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :]
        )  # [b, n-1]
        phi = signed_dihedral_angle(
            C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], C[:, 1:, :]
        )  # [b, n-1]
        bb_angles = torch.stack([psi, omega, phi], dim=-1)  # [b, n-1, 3]

        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        bb_angles = bb_angles * good_pair[..., None]  # [b, n-1, 3]

        zero_pad = torch.zeros((a37.shape[0], 1, 3), device=bb_angles.device)
        bb_angles = torch.cat([bb_angles, zero_pad], dim=1)  # [b, n, 3]
        return bb_angles


class BackboneBondAnglesSeqFeat(Feature):
    """
    Computes bond angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(3 * 21))

    def forward(self, batch):
        bb_bond_angle = self._get_bb_bond_angles(batch)  # [b, n, 3]
        bb_bond_angle_feats = bin_and_one_hot(
            bb_bond_angle,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_bond_angle.device),
        )  # [b, n, 3, nbins]
        bb_bond_angle_feats = einops.rearrange(
            bb_bond_angle_feats, "b n t d -> b n (t d)"
        )  # [b, n, 3 * nbins]
        return bb_bond_angle_feats

    def _get_bb_bond_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]

        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]
        b = a37.shape[0]

        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]
        theta_1 = bond_angles(N[:, :, :], CA[:, :, :], C[:, :, :])  # [b, n]
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]

        # Account for chain breaks in theta_2 and theta_3
        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        theta_2 = theta_2 * good_pair  # [b, n-1]
        theta_3 = theta_3 * good_pair  # [b, n-1]

        # Add a zero at the end of theta_2 and theta_3 to get shape [b, n]
        zero_pad = torch.zeros((b, 1), device=theta_2.device)  # [b, 1]
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)  # [b, n]
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)  # [b, n]

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)  # [b, n, 3]
        return bb_angles


class OpenfoldSideChainAnglesSeqFeat(Feature):
    """Computes sequence features from side chain angles."""

    def __init__(self, **kwargs):
        super().__init__(dim=int(4 * 21 + 4))  # 88

    def forward(self, batch):
        _, angles, torsion_angles_mask = self._get_sidechain_angles(batch)
        # _, [b, n, 4] and [b, n, 4]
        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, 4, nbins]
        angles_feat = angles_feat * torsion_angles_mask[..., None]
        angles_feat = einops.rearrange(
            angles_feat, "b n s d -> b n (s d)"
        )  # [b, n, 4 * nbins]
        feat = torch.cat(
            [angles_feat, torsion_angles_mask], dim=-1
        )  # [b, n, 4 * nbins + 4]
        return feat

    def _get_sidechain_angles(self, batch):
        orig_dtype = batch["coords"].dtype
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"].double()  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"].double()  # [b, n, 37]
        p = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": atom_mask,
        }
        # Next function defined with curry1 decorator
        p = data_transforms.atom37_to_torsion_angles(prefix="")(p)
        torsion_angles_sin_cos = p["torsion_angles_sin_cos"]  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = p["alt_torsion_angles_sin_cos"]  # [b, n, 7, 2]
        # Normalize, all these vectors should have norm 1
        torsion_angles_sin_cos = torsion_angles_sin_cos / (
            torch.linalg.norm(torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos / (
            torch.linalg.norm(alt_torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        torsion_angles_mask = p["torsion_angles_mask"]  # [b, n, 7]
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[..., None]
        alt_torsion_angles_sin_cos = (
            alt_torsion_angles_sin_cos * torsion_angles_mask[..., None]
        )
        angles = torch.atan2(
            torsion_angles_sin_cos[..., 0], torsion_angles_sin_cos[..., 1]
        )  # [b, n, 7]
        angles = angles * torsion_angles_mask
        # Keep only sidechain
        torsion_angles_sin_cos = torsion_angles_sin_cos[..., -4:, :]  # [b, n, 4, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos[
            ..., -4:, :
        ]  # [b, n, 4, 2]
        angles = angles[..., -4:]  # [b, n, 4]
        torsion_angles_mask = torsion_angles_mask[..., -4:]  # [b, n, 4]
        return (
            torsion_angles_sin_cos.to(dtype=orig_dtype),
            angles.to(dtype=orig_dtype),
            torsion_angles_mask.bool(),
        )  # [b, n, 4, 2], [b, n, 4] and [b, n, 4]


class LatentVariableSeqFeat(Feature):
    """Returns sequence feature from latent variable."""

    def __init__(self, latent_z_dim, **kwargs):
        super().__init__(dim=latent_z_dim)

    def forward(self, batch):
        assert (
            "z_latent" in batch
        ), "`z_latent` not in batch, cannot compute LatentVariableSeqFeat"
        return batch["z_latent"]  # [b, n, latent_dim]


class MotifAbsoluteCoordsSeqFeat(Feature):
    """Computes absolute coordinates feature from motif coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for absolute coords
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch:
            batch_coors = {
                "coords_nm": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=False)(batch_coors)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No x_motif or motif_mask in batch, returning zeros for MotifAbsoluteCoordsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifRelativeCoordsSeqFeat(Feature):
    """Computes relative coordinates feature from motif coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for relative coords
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif_mask" in batch:
            required_atoms = torch.tensor(
                [atom_types.index("CA")], device=batch["motif_mask"].device
            )  # CA
            has_required_atoms = torch.all(
                batch["motif_mask"][:, :, required_atoms], dim=-1
            )  # [batch, seq_len]
            relevant_has_required_atoms = torch.where(
                batch["seq_motif_mask"],
                has_required_atoms,
                torch.ones_like(has_required_atoms, dtype=torch.bool),
            )
            if not torch.all(relevant_has_required_atoms):
                if not self._has_logged:
                    logger.warning(
                        "Missing required CA atoms in motif region, returning zeros for MotifRelativeCoordsSeqFeat"
                    )
                    self._has_logged = True
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)
            batch_coors = {
                "coords_nm": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=True)(batch_coors)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No x_motif or motif_mask in batch, returning zeros for MotifRelativeCoordsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifSequenceSeqFeat(Feature):
    """Computes sequence feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=20)  # 20 for one-hot encoded residues
        self._has_logged = False

    def forward(self, batch):
        if "seq_motif" in batch and "seq_motif_mask" in batch:
            batch_seq = {
                "residue_type": batch["seq_motif"],
                "mask_dict": {
                    "residue_type": batch["seq_motif_mask"],
                },
            }
            return ResidueTypeSeqFeat()(batch_seq)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No seq_motif or seq_motif_mask in batch, returning zeros for MotifSequenceSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifSideChainAnglesSeqFeat(Feature):
    """Computes side chain angles feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=88)  # 4 * 21 + 4 for side chain angles
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif" in batch:
            batch_sc_angles = {
                "residue_type": batch["seq_motif"],
                "coords": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return OpenfoldSideChainAnglesSeqFeat()(batch_sc_angles)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "Missing required motif data in batch, returning zeros for MotifSideChainAnglesSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifTorsionAnglesSeqFeat(Feature):
    """Computes torsion angles feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=63)  # 3 * 21 for torsion angles
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif_mask" in batch:
            backbone_atoms = torch.tensor(
                [
                    atom_types.index("N"),
                    atom_types.index("CA"),
                    atom_types.index("C"),
                    atom_types.index("O"),
                ],
                device=batch["motif_mask"].device,
            )
            motif_mask_per_residue_backbone = torch.any(
                batch["motif_mask"][:, :, backbone_atoms], dim=-1
            )  # [batch, seq_len]
            relevant_motif_mask = torch.where(
                batch["seq_motif_mask"],
                motif_mask_per_residue_backbone,
                torch.ones_like(motif_mask_per_residue_backbone, dtype=torch.bool),
            )
            if not torch.all(relevant_motif_mask):
                if not self._has_logged:
                    logger.warning(
                        "Missing backbone atoms in motif region, returning zeros"
                    )
                    self._has_logged = True
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)

            batch_torsion_angles = {
                "coords": batch["x_motif"],
                "residue_pdb_idx": batch.get("residue_pdb_idx", None),
            }
            return BackboneTorsionAnglesSeqFeat()(batch_torsion_angles)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No x_motif or motif_mask in batch, returning zeros for MotifTorsionAnglesSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class XmotifBulkTipSeqFeat(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_seq = ResidueTypeSeqFeat()

        dim = self.const_coors_abs.dim + self.const_seq.dim + 37
        self.dim = dim  # fix dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]

            # concatenate all features
            feat = torch.cat([feat_coors_abs, feat_seq, motif_mask], dim=-1)  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            return feat

        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)


class MotifMaskSeqFeat(Feature):
    """Computes motif mask feature."""

    def __init__(self, **kwargs):
        super().__init__(dim=37)  # 37 for atom mask
        self._has_logged = False

    def forward(self, batch):
        if "motif_mask" in batch:
            return batch["motif_mask"] * 1.0  # [b, n, 37]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No motif_mask in batch, returning zeros for MotifMaskSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class XmotifSeqFeatUnindexed(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""
    
    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_seq = ResidueTypeSeqFeat()
        
        dim = self.const_coors_abs.dim + self.const_seq.dim + 37
        self.dim = dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]
            motif_mask_residue = motif_mask.sum(-1).bool()  # [b, n]

            # concatenate all features
            feat = torch.cat([feat_coors_abs, feat_seq, motif_mask], dim=-1)  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            feats_ind = []
            masks_ind = []
            for b in range(feat.shape[0]):
                feat_local = feat[b, ...]  # [n, some # added up]
                mask_local = motif_mask_residue[b, ...]  # [n]
                feat_local = feat_local[mask_local, ...]  # [# motif residues for b element, some # added up]
                mask_local = mask_local[mask_local]  # [# motif residues for b element], should be all True
                feat_local = feat_local * mask_local[..., None]  # [# motif residues for b element, some # added up]
                feats_ind.append(feat_local)
                masks_ind.append(mask_local)
                assert torch.all(mask_local), "Mask local wrong"

            masks_motif_uidx = pad_sequence(masks_ind, batch_first=True, padding_value=False)  # [b, n]
            feats_motif_uidx = pad_sequence(feats_ind, batch_first=True, padding_value=0.0)  # [b, n, some # added up]

            return feats_motif_uidx, masks_motif_uidx
        
        else:
            raise IOError("No x_motif in batch")
            # b, n = self.extract_bs_and_n(batch)
            # device = self.extract_device(batch)
            # return torch.zeros(b, n, self.dim, device=device)



class BulkAllAtomXmotifSeqFeat(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""
    
    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_coors_rel = Atom37NanometersCoorsSeqFeat(rel=True)
        self.const_seq = ResidueTypeSeqFeat()
        self.const_sc_angles = OpenfoldSideChainAnglesSeqFeat()
        self.const_torsion_angles = BackboneTorsionAnglesSeqFeat()
        
        dim = self.const_coors_abs.dim + self.const_coors_rel.dim + self.const_seq.dim + self.const_sc_angles.dim + self.const_torsion_angles.dim + 37
        self.dim = dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]
            feat_coors_rel = self.const_coors_rel(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # Side chain angles features
            batch_sc_angles = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "coords": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_sc_angles = self.const_sc_angles(batch_sc_angles)  # [b, n, some #]
            if "residue_pdb_idx" in batch:
                idx  = batch["residue_pdb_idx"]  # [b, n]
            else:
                self.assert_defaults_allowed(batch, "Relative sequence separation pair")
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                    device
                )  # [b, n]
            # Torsion angle features
            batch_torsion_angles = {
                "coords": batch["x_motif"],  # [b, n, 37, 3]
                "residue_pdb_idx": idx,  # [b, n]
            }
            feat_torsion_angles = self.const_torsion_angles(batch_torsion_angles)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]

            # concatenate all features
            feat = torch.cat([feat_coors_abs, feat_coors_rel, feat_seq, feat_sc_angles, feat_torsion_angles, motif_mask], dim=-1)  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            return feat
        
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chains" in batch:
            mask = batch["chains"].unsqueeze(-1)  # [b, n, 1]
        else:
            raise ValueError("chains")
        return mask


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert (
            self.dim % 2 == 1
        ), "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` pair dist feature requested but key not available in data modes {data_modes_avail}"
        return bin_pairwise_distances(
            x=batch["x_t"]["bb_ca"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class CaCoorsNanometersPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=30)
        self.min_dist = 0.1
        self.max_dist = 3.0

    def forward(self, batch):
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), f"`ca_coors_nm` pair dist feature requested but key `ca_coors_nm` nor `coords_nm` not available"
        if "ca_coors_nm" in batch:
            ca_coors = batch["ca_coors_nm"]
        else:
            ca_coors = batch["coords_nm"][:, :, 1, :]
        return bin_pairwise_distances(
            x=ca_coors,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class OptionalCaCoorsNanometersPairwiseDistancesPairFeat(
    CaCoorsNanometersPairwiseDistancesPairFeat
):
    """
    If `use_ca_coors_nm_feature` in batch and true, returns pair feature with CA pairwise distances binned, shape [b, n, n, nbins].

    If `use_ca_coors_nm_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if batch.get("use_ca_coors_nm_feature", False):  # defaults to False
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "use_ca_coors_nm_feature disabled or not in batch, returning zeros for OptionalCaCoorsNanometersPairwiseDistancesPairFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class XscBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self,
        x_sc_pair_dist_dim,
        x_sc_pair_dist_min,
        x_sc_pair_dist_max,
        mode_key="x_sc",
        **kwargs,
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "bb_ca" in data_modes_avail
            ), f"`bb_ca` sc/recycle pair dist feature requested but key not available in data modes {data_modes_avail}"
            return bin_pairwise_distances(
                x=batch[self.mode_key]["bb_ca"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {self.mode_key} in batch, returning zeros for XscBBCAPairwiseDistancesPairFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class RelativeResidueOrientationPairFeat(Feature):
    """Computes pair feature with pairwise residue orientations.

    See paper "Improved protein structure prediction using
    predicted inter-residue orientations".

    TODO: Impute beta carbon for Glycine
    TODO: 20 as argument
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(5 * 21))  # 105

    def forward(self, batch):
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"]  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n] boolean
        beta_carbon_pair_mask = (
            has_cb[:, :, None] * has_cb[:, :, None]
        )  # [b, n, n] boolean
        pair_mask = pair_mask * beta_carbon_pair_mask  # [b, n, n]

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        N_p1, CA_p1, CB_p1 = map(
            lambda v: v[:, :, None, :], (N, CA, CB)
        )  # Each [b, n, 1, 3]
        N_p2, CA_p2, CB_p2 = map(
            lambda v: v[:, None, :, :], (N, CA, CB)
        )  # Each [b, 1, n, 3]

        theta_12 = signed_dihedral_angle(N_p1, CA_p1, CB_p1, CB_p2)  # [b, n, n]
        theta_21 = signed_dihedral_angle(N_p2, CA_p2, CB_p2, CB_p1)  # [b, n, n]
        phi_12 = bond_angles(CA_p1, CB_p1, CB_p2)  # [b, n, n]
        phi_21 = bond_angles(CA_p2, CB_p2, CB_p1)  # [b, n, n]
        w = signed_dihedral_angle(CA_p1, CB_p1, CB_p2, CA_p2)  # [b, n, n]
        angles = torch.stack(
            [theta_12, theta_21, phi_12, phi_21, w], dim=-1
        )  # [b, n, n, 5]

        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, n, 5, nbins]
        angles_feat = einops.rearrange(
            angles_feat, "b n m f d -> b n m (f d)"
        )  # [b, n, n, 5 * nbins]
        angles_feat = angles_feat * pair_mask[..., None]  # Mask padding
        return angles_feat


class BackbonePairDistancesNanometerPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms.

    Position (i, j) encodes the distance between CA_i and
    {N_j, CA_j, C_j, CB_j}.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(4 * 21))  # 84

    def forward(self, batch):
        assert (
            "coords_nm" in batch
        ), "`coords_nm` not in batch, cannot comptue BackbonePairDistancesNanometerPairFeat"
        coords = batch["coords_nm"]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        C = coords[:, :, 2, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        CA_i = CA[:, :, None, :]  # [b, n, 1, 3]
        N_j, CA_j, C_j, CB_j = map(
            lambda v: v[:, None, :, :], (N, CA, C, CB)
        )  # Each [b, 1, n, 3]

        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: torch.linalg.norm(v[0] - v[1], dim=-1),
            ((CA_i, N_j), (CA_i, CA_j), (CA_i, C_j), (CA_i, CB_j)),
        )  # Each shape [b, n, n]
        # CA_X[..., i, j] has distance (nm) between CA[..., i] and X[..., j]

        # Accomodate residues without CB
        CA_CB = CA_CB * has_cb[:, None, :]  # [b, n, n]

        # Fix for mask
        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: v * pair_mask,
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each shape [b, n, n]

        bin_limits = torch.linspace(0.1, 2, 20, device=coords.device)
        CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat = map(
            lambda v: bin_and_one_hot(v, bin_limits=bin_limits),
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each [b, n, n, 21]

        feat = torch.cat(
            [CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat], dim=-1
        )  # [b, n, n, 4 * 21]
        feat = feat * pair_mask[..., None]
        return feat


class XmotifPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone motif atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.const = BackbonePairDistancesNanometerPairFeat()
        self.dim = self.const.dim  # Fix dim, cannot put init here
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch:
            batch_bbpd = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat = self.const(batch_bbpd)  # [b, n, n, some #]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No x_motif in batch, returning zeros for XmotifPairwiseDistancesPairFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class ChainIdxPairFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chains" in batch:
            seq_mask = batch["chains"]  # [b, n]
            mask = torch.einsum("bi,bj->bij", seq_mask, seq_mask).unsqueeze(
                -1
            )  # [b, n, n, 1]
        else:
            raise ValueError("chains")
        return mask


class StochasticTranslationSeqFeat(Feature):
    """Gets stochastic translation from centering transform and returns feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)
        self._has_logged = False

    def forward(self, batch):
        if "stochastic_translation" in batch:
            b, n = self.extract_bs_and_n(batch)
            translation = batch["stochastic_translation"]  # [b, 3]
            # Broadcast translation to all residues
            mask = translation[:, None, :].expand(b, n, -1)  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "No stochastic_translation in batch, returning zeros for StochasticTranslationSeqFeat"
                )
                self._has_logged = True
            mask = torch.zeros((b, n, 3), device=device)
        return mask


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        **kwargs,
    ):
        """
        Feature factory for creating sequence and pair features.

        Sequence features include:
            Time embeddings:
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Position and structure:
                - "res_seq_pdb_idx": Residue sequence position (requires ResidueSequencePositionPdbTransform)
                - "chain_break_per_res": Chain break per residue (requires ChainBreakPerResidueTransform)
                - "fold_emb": Fold embedding

            Coordinates and angles:
                - "x_sc_bb_ca": Self-conditioning backbone CA coordinates
                - "x_recycle_bb_ca": Recycled backbone CA coordinates
                - "x_sc_local_latents": Self-conditioning local latents
                - "x_recycle_local_latents": Recycled local latents
                - "xt_bb_ca": Target backbone CA coordinates
                - "xt_local_latents": Target local latents
                - "ca_coors_nm": CA coordinates in nanometers
                - "ca_coors_nm_try": Try CA coordinates in nanometers
                - "optional_ca_coors_nm_seq_feat": Optional CA coordinates in nanometers
                - "x1_bb_angles": Backbone torsion angles
                - "x1_bond_angles": Backbone bond angles
                - "x1_sidechain_angles": Sidechain angles

            Residue information:
                - "x1_aatype": Residue type
                - "optional_res_type_seq_feat": Optional residue type
                - "x1_a37coors_nm": Atom37 coordinates in nanometers
                - "x1_a37coors_nm_rel": Relative atom37 coordinates in nanometers

            Motif and target:
                - "x_motif": Motif coordinates and features
                - "z_latent_seq": Latent variable sequence

        Pair features include:
            Distance features:
                - "xt_bb_ca_pair_dists": Target backbone CA pairwise distances
                - "x_sc_bb_ca_pair_dists": Self-conditioning backbone CA pairwise distances
                - "x_recycle_bb_ca_pair_dists": Recycled backbone CA pairwise distances
                - "ca_coors_nm_pair_dists": CA coordinates pairwise distances in nanometers
                - "x1_bb_pair_dists_nm": Backbone pairwise distances in nanometers
                - "optional_ca_pair_dist": Optional CA pairwise distances
                - "x_motif_pair_dists": Motif pairwise distances

            Sequence and time:
                - "rel_seq_sep": Relative sequence separation
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Structure and orientation:
                - "x1_bb_pair_orientation": Relative residue orientation
                - "chain_idx_pair": Chain index pairwise feature

        """
        super().__init__()
        self.mode = mode
        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList(
            [self.get_creator(f, **kwargs) for f in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False
        )

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingSeqFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingSeqFeat(data_mode_use="local_latents", **kwargs)

            # Position and indexing
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "chain_idx_seq":
                return ChainIdxSeqFeat(**kwargs)
            elif f == "fold_emb":
                return FoldEmbeddingSeqFeat(**kwargs)
            elif f == "cropped_flag_seq":
                return CroppedFlagSeqFeat()


            # Basic residue information
            elif f == "x1_aatype":
                return ResidueTypeSeqFeat(**kwargs)
            elif f == "optional_res_type_seq_feat":
                return OptionalResidueTypeSeqFeat(**kwargs)

            # Raw coordinate features
            elif f == "ca_coors_nm":
                return CaCoorsNanometersSeqFeat(**kwargs)
            elif f == "ca_coors_nm_try":
                return TryCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "optional_ca_coors_nm_seq_feat":
                return OptionalCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm":
                return Atom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm_rel":
                return Atom37NanometersCoorsSeqFeat(rel=True, **kwargs)

            # Diffusion/sampling coordinates
            elif f == "xt_bb_ca":
                return XtBBCASeqFeat(**kwargs)
            elif f == "xt_local_latents":
                return XtLocalLatentsSeqFeat(**kwargs)
            elif f == "x_sc_bb_ca":
                return XscBBCASeqFeat(**kwargs)
            elif f == "x_recycle_bb_ca":
                return XscBBCASeqFeat(mode_key="x_recycle", **kwargs)
            elif f == "x_sc_local_latents":
                return XscLocalLatentsSeqFeat(**kwargs)
            elif f == "x_recycle_local_latents":
                return XscLocalLatentsSeqFeat(mode_key="x_recycle", **kwargs)

            # Structural features (angles)
            elif f == "x1_bb_angles":
                return BackboneTorsionAnglesSeqFeat(**kwargs)
            elif f == "x1_bond_angles":
                return BackboneBondAnglesSeqFeat(**kwargs)
            elif f == "x1_sidechain_angles":
                return OpenfoldSideChainAnglesSeqFeat(**kwargs)

            # Latent variables
            elif f == "z_latent_seq":
                return LatentVariableSeqFeat(**kwargs)

            # Motif features
            elif f == "motif_abs_coords":
                return MotifAbsoluteCoordsSeqFeat(**kwargs)
            elif f == "motif_rel_coords":
                return MotifRelativeCoordsSeqFeat(**kwargs)
            elif f == "motif_seq":
                return MotifSequenceSeqFeat(**kwargs)
            elif f == "motif_sc_angles":
                return MotifSideChainAnglesSeqFeat(**kwargs)
            elif f == "motif_torsion_angles":
                return MotifTorsionAnglesSeqFeat(**kwargs)
            elif f == "motif_mask":
                return MotifMaskSeqFeat(**kwargs)
            
            #######################################################
            # Some manual changes needed here for the motif task
            #######################################################

            # # Uncomment this for indexed + tip atom
            # elif f == "x_motif":
            #     return XmotifBulkTipSeqFeat(**kwargs)

            # Uncomment this for any other motif task
            elif f == "bulk_all_atom_xmotif" or f == "x_motif":
                return BulkAllAtomXmotifSeqFeat(**kwargs)
            
            #######################################################
            #######################################################
            #######################################################

            elif f == "bulk_all_atom_xmotif":
                return BulkAllAtomXmotifSeqFeat(**kwargs)

            # Centering
            elif f == "stochastic_translation":
                return StochasticTranslationSeqFeat(**kwargs)

            # Special/utility features
            elif f == "zero_feat_seq":
                return ZeroFeat(**kwargs)
            else:
                raise IOError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingPairFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingPairFeat(data_mode_use="local_latents", **kwargs)

            # Sequence separation
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)

            # Distance features
            elif f == "xt_bb_ca_pair_dists":
                return XtBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_sc_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_recycle_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(mode_key="x_recycle", **kwargs)
            elif f == "ca_coors_nm_pair_dists":
                return CaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "optional_ca_pair_dist":
                return OptionalCaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "x1_bb_pair_dists_nm":
                return BackbonePairDistancesNanometerPairFeat(**kwargs)
            elif f == "x_motif_pair_dists":
                return XmotifPairwiseDistancesPairFeat(**kwargs)

            # Structural and orientation features
            elif f == "x1_bb_pair_orientation":
                return RelativeResidueOrientationPairFeat(**kwargs)

            # Chain and indexing features
            elif f == "chain_idx_pair":
                return ChainIdxPairFeat(**kwargs)

            else:
                raise IOError(f"Pair feature {f} not implemented.")

        else:
            raise IOError(
                f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'."
            )

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(
                fcreator(batch)
            )  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(
            feature_tensors, dim=-1
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(
            features, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(
            self.linear_out(features)
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(
            features_proc, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]


class FeatureFactoryUidxMotif(torch.nn.Module):
    def __init__(
        self,
        # feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        **kwargs,
    ):
        """
        Sequence features include:
            - "x_motix_uidx"

        Pair features include:
        """
        super().__init__()

        self.feat_creator = XmotifSeqFeatUnindexed(**kwargs)
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(self.feat_creator.get_dim(), dim_feats_out, bias=False)

    def forward(self, batch):
        """Returns feature and mask, shapes [b, n_motif, dim_f] and [b, n_motif]"""
        feat, feat_mask = self.feat_creator(batch)  # [b, n_motif, dim_f], [b, n_motif]
        feat_proc = self.ln_out(self.linear_out(feat))  # [b, n_motif, dim_f]
        return feat_proc * feat_mask[..., None], feat_mask

