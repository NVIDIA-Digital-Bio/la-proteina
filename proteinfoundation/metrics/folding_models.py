import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from biotite.structure.io import pdb, pdbx
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

hf_logging.set_verbosity_error()


def create_individual_fasta_files(
    sequences: List[str],
    output_dir: str,
    format_type: Literal["simple", "chai1", "boltz2"] = "simple",
    name_prefix: str = "seq",
) -> str:
    """
    Creates individual FASTA files for each sequence with appropriate headers for different folding models.

    Args:
        sequences: List of protein sequences
        output_dir: Directory where individual FASTA files will be written
        format_type: Header format to use:
            - "simple": >seq_1, >seq_2, ... (for ESMFold/ColabFold)
            - "chai1": >protein|name=seq_1, >protein|name=seq_2, ... (for Chai-1)
            - "boltz2": A|protein|seq_1, A|protein|seq_2, ... (for Boltz2)
        name_prefix: Prefix for sequence names

    Returns:
        Path to the directory containing the individual FASTA files
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, seq in enumerate(sequences):
        seq_name = f"{name_prefix}_{i+1}"
        fasta_path = os.path.join(output_dir, f"{seq_name}.fasta")

        if format_type == "simple":
            header = f">{seq_name}"
        elif format_type == "chai1":
            header = f">protein|name={seq_name}"
        elif format_type == "boltz2":
            header = f">A|protein|empty"
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        with open(fasta_path, "w") as f:
            f.write(f"{header}\n{seq}\n")

    return output_dir


def run_esmfold(
    sequences: List[str],
    path_to_esmfold_out: str,
    name: str,
    suffix: str,
    cache_dir: Optional[str] = None,
    keep_outputs: bool = False,
) -> List[str]:
    """
    Runs ESMFold on sequences and stores results as PDB files.

    For now, runs with a single GPU, though not a big deal if we parallelie jobs (easily
    done with our inference pipeline).

    Args:
        sequences: List of protein sequences to predict
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs
        name: name to use when storing
        suffix: to use as suffix when storing files
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep output directories

    Returns:
        List of paths (list of str) to PDB files
    """
    is_cluster_run = os.environ.get("SLURM_JOB_ID") is not None

    # Use provided cache_dir or fallback to environment/cluster logic
    final_cache_dir = cache_dir
    if final_cache_dir is None and is_cluster_run:
        final_cache_dir = os.environ.get("CACHE_DIR")

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1", cache_dir=final_cache_dir
    )
    esm_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=final_cache_dir
    )
    esm_model = esm_model.cuda()

    # Run ESMFold
    list_of_strings_pdb = []
    if len(sequences) == 8:
        max_nres = max([len(x) for x in sequences])
        if max_nres > 700:
            batch_size = 1
            num_batches = 8
        elif max_nres > 500:
            batch_size = 2
            num_batches = 4
        elif max_nres > 200:
            batch_size = 4
            num_batches = 2
        else:
            batch_size = 8
            num_batches = 1
    elif len(sequences) == 1:
        batch_size = 8
        num_batches = 1
    else:
        raise IOError(
            "We can only run ESMFold with 1 or 8 sequences... We should fix this..."
        )

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        inputs = tokenizer(
            sequences[start_idx:end_idx],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {k: inputs[k].cuda() for k in inputs}

        with torch.no_grad():
            _outputs = esm_model(**inputs)

        _list_of_strings_pdb = _convert_esm_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)

    # Create out directory if not there
    if not os.path.exists(path_to_esmfold_out):
        os.makedirs(path_to_esmfold_out)

    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        fname = f"esm_{i+1}.pdb_esm_{suffix}"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
            out_esm_paths.append(fdir)

    if not keep_outputs:
        # Clean up individual FASTA files directory
        try:
            shutil.rmtree(os.path.dirname(os.path.dirname(fdir)))
        except Exception as e:
            logger.warning(f"Could not clean up FASTA directory: {e}")

    return out_esm_paths


# I got this function from hugging face's ESM notebook example
def _convert_esm_outputs_to_pdb(outputs) -> List[str]:
    """Takes ESMFold outputs and converts them to a list of PDBs (as strings)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs
