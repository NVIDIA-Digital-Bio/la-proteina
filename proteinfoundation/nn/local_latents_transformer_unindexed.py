from typing import Dict

import torch

from openfold.np.residue_constants import RESTYPE_ATOM37_MASK
from proteinfoundation.nn.feature_factory import FeatureFactory, FeatureFactoryUidxMotif
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.modules.pair_rep_initial import PairReprBuilder


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(
        dtype=torch.bool, device=device
    )  # [21, 37]


class LocalLatentsTransformerMotifUidx(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(LocalLatentsTransformerMotifUidx, self).__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.motif_uidx_factory = FeatureFactoryUidxMotif(
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

    # @torch.compile
    def forward(self, input: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # unindexed motif
        motif_uidx, motif_mask = self.motif_uidx_factory(input)  # [b, nres_motif, token_dim], [b, nres_motif]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_repr_builder(input)  # [b, n, n, pair_dim]

        # Extend everything: concat motif to sequence, extend cond with zeros, extend mask with motif, extend pair_rep with zeros
        b, n_orig, _ = seqs.shape
        _, n_motif, _ = motif_uidx.shape
        dim_pair = pair_rep.shape[-1]
        dim_cond = c.shape[-1]

        # Extend sequence rep and conditioning
        seqs = torch.cat([seqs, motif_uidx], dim=1)  # [b, n + nres_motif, token_dim]
        zero_tensor = torch.zeros(b, n_motif, dim_cond, device=seqs.device)  # [b, n_motif, dim_cond]
        c = torch.cat([c, zero_tensor], dim=1)  # [b, n + nres_motif, dim_cond]

        # Extend mask
        mask = torch.cat([mask, motif_mask], dim=1)  # [b, n + nres_motif]

        # Extend pair representation with zeros; pair has shape [b, n, n, pair_dim] -> [b, n+nres_motif, n+nres_motif, pair_dim]
        # [b, n, n, pair_dim] -> [b, n + nres_motif, n, pair_dim]
        zero_pad_d1 = torch.zeros(
            b, n_motif, n_orig, dim_pair, device=seqs.device
        )  # [b, n_motif, n_orig, dim_pair]
        pair_rep = torch.cat([pair_rep, zero_pad_d1], dim=1)  # [b, n + nres_motif, n, dim_pair]
        # [b, n + nres_motif, n, pair_dim] -> [b, n + nres_motif, n + nres_motif, pair_dim]
        zero_pad_d2 = torch.zeros(
            b, n_orig + n_motif, n_motif, dim_pair, device=seqs.device
        )  # [b, n_orig + n_motif, n_orig, dim_pair]
        pair_rep = torch.cat([pair_rep, zero_pad_d2], dim=2)  # [b, n_orig + n_motif, n_orig + n_motif, dim_pair]

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, pair_dim]

        # Get outputs
        local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n, 3]

        # Unextend
        local_latents_out = local_latents_out[:, :n_orig, :]
        ca_nm_out = ca_nm_out[:, :n_orig, :]
        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {
            self.output_param["local_latents"]: local_latents_out
        }
        return nn_out
