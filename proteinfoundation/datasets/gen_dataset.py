import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from proteinfoundation.utils.align_utils import mean_w_mask
from proteinfoundation.utils.fold_utils import mask_cath_code_by_level
from proteinfoundation.utils.motif_utils import parse_motif, save_motif_csv
from proteinfoundation.utils.coors_utils import ang_to_nm


class GenDataset(Dataset):
    """
    This class provides length-centric and fold-centric sampling for unconditional
    and conditional protein structure generation. Each returned item is a dictionary
    with key information for generation, which contains the length of proteins, the
    number of proteins and cath codes if conditional sampling is used.

    If length distribution is specified, sample `nsamples` proteins for each length,
    cath codes are randomly sampled based on empirical distribution.
    Otherwise, if cath code set is specified, sample `nsamples` proteins for each cath code,
    lengths are randomly sampled based on empirical distribution.

    Each sample returned by this dataset is a 2-tuple (L, nsamples) or 3-tuple (L, nsamples, cath_code) where
      - nres (int) is the number of residues in the proteins to be samples
      - nsamples (int) is the number of proteins to generate (happens in parallel),
        so if nsamples=10 it means that it will produce 10 proteins of length L (all sampled in parallel)
      - cath_code (List[str]) is the cath code for the nsamples if conditional generation is used
    """

    def __init__(
        self,
        nlens_cfg: Optional[Dict] = None,
        cath_codes: Optional[List[str]] = None,
        empirical_distribution_cfg: Optional[Dict] = None,
        motif_dict_cfg: Optional[Dict] = None,
        motif_task_name: Optional[str] = None,
        motif_csv_path: Optional[str] = None,
        target_as_features: bool = False,
        nsamples: Optional[int] = 1,
        max_nsamples_per_batch: Optional[int] = 1,
        n_replicas: int = 1,
    ):
        """
        Args:
            nlens_cfg (Optional[Dict]): Config dict for specifying length distribution. If not None, use length-centric sampling.
                Defaults to None.
            cath_codes (Optional[List[str]]): List of cath codes to sample.. If not None and nlens_cfg is None, use fold-centric sampling.
                Defaults to None.
            empirical_distribution_cfg (Optional[Dict]): Config dict for specifying (length, cath code) distribution.
                Defaults to None.
            motif_dict_cfg (Optional[Dict]): Config dict for all motif configs.
                Defaults to None.
            motif_task_name (Optional[str]): Name of the motif task to sample.
                Defaults to None.
            motif_csv_path (Optional[str]): Path to the motif csv file.
                Defaults to None.

            nsamples (Optional[int]): Number of samples to generate for each length or each cath code.
                Defaults to 1.
            max_nsamples_per_batch (Optional[int]): Maximum number of samples for each batch.
                Defaults to 1.
            n_replicas (Optional[int]): Number of devices. Used for validation on multiple devices.
                Defaults to 1.
        """
        super(GenDataset, self).__init__()
        ##################################################################################
        ################### 1. Parse length and cath codes ###############################
        ##################################################################################
        nres = self.parse_nlens_cfg(nlens_cfg)
        self.target_as_features = target_as_features
        self.motif_task_name = motif_task_name
        if nres is not None:
            logger.info("Use length-centric sampling.")
            nsamples = [nsamples] * len(nres)
        elif motif_task_name:
            logger.info("Use motif-conditioned sampling.")
            if motif_task_name in motif_dict_cfg:
                motif_cfg = motif_dict_cfg[motif_task_name]
            else:
                raise ValueError(
                    f"Motif task name {motif_task_name} not found in motif_dict_cfg"
                )
            nsamples = [nsamples]
        else:
            raise ValueError("Error in GenDataset init.")

        ##################################################################################
        ################### 2. Parse and bucketize empirical distribution ################
        ##################################################################################
        if empirical_distribution_cfg:
            self.parse_empirical_distribution_cfg(empirical_distribution_cfg)
            self.bucketize()

        ##################################################################################
        ################### 3. Generate data points ######################################
        ##################################################################################
        self.motif_masks = [None] * len(nsamples)
        self.x_motifs = [None] * len(nsamples)
        self.masks = [None] * len(nsamples)
        if nres is not None:
            # Length-centric generation
            self.nres, self.cath_codes, self.nsamples = (
                self.generate_cath_code_given_len(nres, nsamples)
            )
        elif cath_codes:
            # Fold-centric generation
            self.nres, self.cath_codes, self.nsamples = (
                self.generate_len_given_cath_code(cath_codes, nsamples)
            )
        else:
            self.nsamples = nsamples
            self.cath_codes = [None] * len(nsamples)
            self.motif_masks, self.x_motifs, self.residue_types = (
                self.generate_motif_info(motif_cfg, nsamples[0], motif_csv_path)
            )

        ##################################################################################
        # 4. Make sure the nsamples for each data point is not greater than max_nsamples #
        ##################################################################################
        if max_nsamples_per_batch:
            if nres is not None or cath_codes:
                self.nres, self.cath_codes, self.nsamples = self.flatten(
                    max_nsamples_per_batch
                )
            else:
                (
                    self.nres,
                    self.cath_codes,
                    self.nsamples,
                    self.masks,
                    self.motif_masks,
                    self.x_motifs,
                    self.residue_types,
                ) = self.flatten_motif(max_nsamples_per_batch)

        ##################################################################################
        # 5. Make sure this won't cause an error during validation on multiple devices ###
        ##################################################################################
        if n_replicas > 1:
            self.pad_nlens(n_replicas)
        assert all(
            [n <= max_nsamples_per_batch for n in self.nsamples]
        ), f"The nsamples for each len shouldn't be greater than {max_nsamples_per_batch}"
        assert (
            len(self.nsamples) % n_replicas == 0
        ), f"Should be evenly splitable over {n_replicas} devices"

        logger.info(
            f"Adding generation dataset to sample {self.nsamples} sequences of length {self.nres}."
        )

    def bucketize(self):
        """Build length buckets for cath_codes. Record the cath_code distribution given length bucket and the reverse"""
        if self.len_cath_codes is None:
            self.cath_codes_given_len_bucket = None
            self.len_bucket_given_cath_codes = None
            return

        bucket = list(
            range(self.bucket_min_len, self.bucket_max_len, self.bucket_step_size)
        )
        cath_codes_given_len_bucket = [[] for _ in range(len(bucket))]
        len_bucket_given_cath_codes = defaultdict(set)
        for _len, codes in self.len_cath_codes:
            if len(codes) == 0:
                continue
            bucket_idx = (_len - self.bucket_min_len) // self.bucket_step_size
            bucket_idx = min(bucket_idx, self.bucket_size - 1)  # Boundary cutoff
            bucket_idx = max(bucket_idx, 0)

            # Record all possible cath codes for each bucket
            cath_codes_given_len_bucket[bucket_idx].append(codes)

            # Record all possible len bucket for each cath code
            for code in codes:
                for level in ["C", "A", "T"]:
                    ns = {"C": 3, "A": 2, "T": 1}
                    level_code = code.rsplit(".", ns[level])[0] + ".x" * ns[level]
                    len_bucket_given_cath_codes[level_code].add(bucket_idx)

        for k, v in len_bucket_given_cath_codes.items():
            len_bucket_given_cath_codes[k] = tuple(v)

        self.cath_codes_given_len_bucket = cath_codes_given_len_bucket
        self.len_bucket_given_cath_codes = len_bucket_given_cath_codes

    def generate_cath_code_given_len(self, nres: List[int], nsamples: List[int]):
        """Pre-generate corresponding cath codes for each length"""
        cath_codes = []
        for i in range(len(nres)):
            if self.cath_codes_given_len_bucket is None:
                cath_code = None
            else:
                if nres[i] <= self.bucket_max_len:
                    bucket_idx = (
                        nres[i] - self.bucket_min_len
                    ) // self.bucket_step_size
                else:
                    bucket_idx = -1
                cath_code = random.choices(
                    self.cath_codes_given_len_bucket[bucket_idx], k=nsamples[i]
                )
            cath_codes.append(cath_code)
        return nres, cath_codes, nsamples

    def generate_len_given_cath_code(self, cath_codes: List[str], nsamples: List[int]):
        """Pre-generate corresponding lengths for each cath code, then gather proteins of the same length as one batch"""
        assert (
            self.len_bucket_given_cath_codes is not None
        ), "Need len_cath_code distribution for fold-centric generation"
        tmp_nres = []
        tmp_cath_codes = []
        for i in range(len(cath_codes)):
            for _ in range(nsamples[i]):
                if cath_codes[i] not in self.len_bucket_given_cath_codes:
                    raise ValueError(
                        f"CATH code {cath_codes[i]} not in the empirical distribution"
                    )
                bucket_idx = random.choices(
                    self.len_bucket_given_cath_codes[cath_codes[i]], k=1
                )[0]
                _len = self.bucket_min_len + bucket_idx * self.bucket_step_size

                tmp_nres.append(_len)
                tmp_cath_codes.append([cath_codes[i]])

        # Gather the same lengths, as we need to generate proteins of the same length together
        len_bucket = defaultdict(list)
        out_nres, out_cath_codes, out_nsamples = [], [], []
        for n, code in zip(tmp_nres, tmp_cath_codes):
            len_bucket[n].append(code)

        for n, code in len_bucket.items():
            out_nres.append(n)
            out_cath_codes.append(code)
            out_nsamples.append(len(code))

        return out_nres, out_cath_codes, out_nsamples

    def generate_motif_info(self, motif_cfg, nsamples, motif_csv_path):
        # Always return motif_masks, x_motifs, residue_types as lists of tensors, regardless of input type
        lengths, motif_masks, x_motifs, residue_types, outstrs = parse_motif(
            nsamples=nsamples, **motif_cfg
        )
        idx = np.argsort(lengths)
        motif_masks = [motif_masks[i] for i in idx]
        x_motifs = [x_motifs[i] for i in idx]
        residue_types = [residue_types[i] for i in idx]
        # center motifs to origin
        for i in range(len(x_motifs)):
            motif_center = mean_w_mask(
                x_motifs[i].flatten(0, 1), motif_masks[i].flatten(0, 1)
            ).unsqueeze(0)
            x_motifs[i] = x_motifs[i] - motif_center
            x_motifs[i] = x_motifs[i] * motif_masks[i][..., None]
        # Only save CSV for contig_string (residue/range) case
        if "motif_atom_spec" not in motif_cfg or motif_cfg["motif_atom_spec"] is None:
            outstrs = [outstrs[i] for i in idx]
            save_motif_csv(
                motif_cfg["motif_pdb_path"],
                self.motif_task_name,
                outstrs,
                outpath=motif_csv_path,
                segment_order=motif_cfg["segment_order"],
            )
        return motif_masks, x_motifs, residue_types

    def flatten(self, max_nsamples: int):
        """Flatten the list to make sure each data point have no more than max_nsamples"""
        nres, cath_codes, nsamples = [], [], []
        for i in range(len(self.nsamples)):
            for j in range(0, self.nsamples[i], max_nsamples):
                nres.append(self.nres[i])
                if self.cath_codes[i] is not None:
                    cath_codes.append(self.cath_codes[i][j : j + max_nsamples])
                else:
                    cath_codes.append(None)
                if j + max_nsamples <= self.nsamples[i]:
                    nsamples.append(max_nsamples)
                else:
                    nsamples.append(self.nsamples[i] - j)
        return nres, cath_codes, nsamples

    def flatten_motif(self, max_nsamples: int):
        """Flatten the list to make sure each data point have no more than max_nsamples"""
        nres, cath_codes, nsamples = [], [], []
        masks, motif_masks = [], []
        x_motifs, residue_types = [], []
        for i in range(len(self.nsamples)):
            for j in range(0, self.nsamples[i], max_nsamples):

                if self.cath_codes[i] is not None:
                    cath_codes.append(self.cath_codes[i][j : j + max_nsamples])
                else:
                    cath_codes.append(None)
                if j + max_nsamples <= self.nsamples[i]:
                    nsamples.append(max_nsamples)
                    motif_mask = self.motif_masks[j : j + max_nsamples]
                    x_motif = self.x_motifs[j : j + max_nsamples]
                    residue_type = self.residue_types[j : j + max_nsamples]
                else:
                    nsamples.append(self.nsamples[i] - j)
                    motif_mask = self.motif_masks[j : self.nsamples[i]]
                    x_motif = self.x_motifs[j : self.nsamples[i]]
                    residue_type = self.residue_types[j : self.nsamples[i]]
                mask = [torch.Tensor([True] * x.shape[0]) for x in motif_mask]
                padded_mask = torch.nn.utils.rnn.pad_sequence(
                    mask, batch_first=True, padding_value=False
                )
                padded_motif_mask = torch.nn.utils.rnn.pad_sequence(
                    motif_mask, batch_first=True, padding_value=False
                )
                padded_x_motif = torch.nn.utils.rnn.pad_sequence(
                    x_motif, batch_first=True, padding_value=0
                )
                padded_residue_type = torch.nn.utils.rnn.pad_sequence(
                    residue_type, batch_first=True, padding_value=0
                )
                masks.append(padded_mask)
                motif_masks.append(padded_motif_mask)
                x_motifs.append(padded_x_motif)
                residue_types.append(padded_residue_type)
                nres.append(padded_mask.shape[1])
        return nres, cath_codes, nsamples, masks, motif_masks, x_motifs, residue_types

    def pad_nlens(self, n_replicas: int):
        """Split nlens into data points (len, nsample) as val dataset and guarantee that
        1. len(val_dataset) should be a multiple of n_replica, to ensure that we don't introduce additional samples for multi-gpu validation
        2. nsample should be the same for all data points if n_replica > 1 (multi-gpu)
        """
        # Add samples to the small bins
        max_nsamples = max(self.nsamples)
        for i in range(len(self.nsamples)):
            while self.cath_codes[i] != None and len(self.cath_codes[i]) < max_nsamples:
                self.cath_codes[i] += self.cath_codes[i][
                    : (max_nsamples - len(self.cath_codes[i]))
                ]
            self.nsamples[i] += max_nsamples - self.nsamples[i]

        # Keep adding lengths in the dataset to make it a multiple of n_replica
        while len(self.nres) % n_replicas != 0:
            self.nres.append(self.nres[-1])
            self.nsamples.append(max_nsamples)
            self.cath_codes.append(self.cath_codes[-1])
            self.cath_codes.append(self.cath_codes[-1])
            if hasattr(self, "chain_masks"):
                self.chain_masks.append(self.chain_masks[-1])
            if hasattr(self, "general_masks"):
                self.general_masks.append(self.general_masks[-1])
            if hasattr(self, "structures"):
                self.structures.append(self.structures[-1])

    def parse_empirical_distribution_cfg(self, cfg: Dict):
        """Load empirical (len, cath_codes) joint distribution. Apply mask according to the guidance cath code level"""
        if cfg.len_cath_code_path is not None:
            logger.info(
                f"Loading empirical (length, cath_code) distribution from {cfg.len_cath_code_path}"
            )
            raw_len_cath_codes = torch.load(cfg.len_cath_code_path)

            # By applying mask to the cath code distribution, we can control the level we want to sample
            level = cfg.cath_code_level
            self.len_cath_codes = []
            for i in range(len(raw_len_cath_codes)):
                _len, code = raw_len_cath_codes[i]
                code = mask_cath_code_by_level(code, level="H")
                if level == "A" or level == "C":
                    code = mask_cath_code_by_level(code, level="T")
                    if level == "C":
                        code = mask_cath_code_by_level(code, level="A")
                self.len_cath_codes.append((_len, code))

            self.bucket_min_len = cfg.bucket_min_len
            self.bucket_max_len = cfg.bucket_max_len
            self.bucket_step_size = cfg.bucket_step_size
            self.bucket_size = (
                self.bucket_max_len - self.bucket_min_len
            ) // self.bucket_step_size + 1
        else:
            logger.info(
                "No empirical (length, cath_code) distribution provided. Use unconditional training."
            )
            self.len_cath_codes = None

    def parse_nlens_cfg(self, cfg: Dict):
        """Load nlens config."""
        if cfg is None:
            return None
        if cfg.nres_lens:
            nres = [int(n) for n in cfg.nres_lens]
        elif cfg.min_len:
            nres = np.arange(cfg.min_len, cfg.max_len + 1, cfg.step_len).tolist()
        else:
            nres = None
        return nres

    def __len__(self):
        return len(self.nres)

    def __getitem__(self, index: int):
        result = {
            "nres": self.nres[index],
            "nsamples": self.nsamples[index],
        }

        # Add CATH codes if available
        if self.cath_codes[index] is not None:
            result["cath_code"] = self.cath_codes[index]
        # Motif-conditioned
        if self.motif_task_name is not None:
            # Assume motif_mask, x_motif, seq_motif_mask, seq_motif are available
            result["motif_mask"] = self.motif_masks[index].bool()  # [bs, num_res, 37]
            result["x_motif"] = self.x_motifs[index]  # [bs, num_res, 37, 3]
            result["seq_motif_mask"] = (
                self.motif_masks[index].sum(dim=-1).bool()
            )  # [bs, num_res]
            result["seq_motif"] = self.residue_types[index]  # [bs, num_res]
            result["mask"] = self.masks[index].bool()  # [bs, num_res]
            return result

        # Fallback: unconditional
        return result
