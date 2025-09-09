import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from biotite.sequence.io import fasta
from jaxtyping import Bool, Float
from loguru import logger
from torch import Tensor
from transformers import logging as hf_logging

from openfold.np import residue_constants
from proteinfoundation.metrics.folding_models import run_esmfold
from proteinfoundation.utils.align_utils import kabsch_align_ind
from proteinfoundation.utils.coors_utils import (
    get_atom37_bb3_mask,
    get_atom37_bb3o_mask,
    get_atom37_ca_mask,
)
from proteinfoundation.utils.pdb_utils import load_pdb

hf_logging.set_verbosity_error()


def pdb_name_from_path(pdb_file_path: str) -> str:
    """Extracts the PDB filename without extension from a file path.

    Args:
        pdb_file_path: Full path to the PDB file.

    Returns:
        The PDB filename without the .pdb extension.
    """
    return pdb_file_path.strip(os.sep).split(os.sep)[-1][
        :-4
    ]  # Name of the pdb file without ".pdb" extension


# ProteinMPNN
## ## ## ## ## ## ## ## ## ## ## ##


def extract_gen_seqs(path_to_file: str) -> List[Dict[str, float]]:
    """Extracts sequences and metadata from ProteinMPNN generation files.

    Args:
        path_to_file: Path to file with ProteinMPNN output in FASTA format.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value
    """
    seqs = []
    fasta_file = fasta.FastaFile.read(path_to_file)

    # Skip first sequence (model info)
    first = True
    for header, sequence in fasta_file.items():
        if first:
            first = False
            continue

        # Extract score and seq_recovery from header
        score_match = re.search(r"score=([\d\.]+)", header)
        seqid_match = re.search(r"seq_recovery=([\d\.]+)", header)

        if score_match and seqid_match:
            seqs.append(
                {
                    "seq": sequence,
                    "score": float(score_match.group(1)),
                    "seqid": float(seqid_match.group(1)),
                }
            )

    return seqs


def run_proteinmpnn(
    pdb_file_path: str,
    out_dir_root: str,
    all_chains: List[str] = ["A"],
    pdb_path_chains: List[str] = ["A"],
    fix_pos: List[str] = None,
    num_seq_per_target: int = 8,
    omit_AAs: List[str] = None,
    sampling_temp: float = 0.1,
    seed: Optional[int] = None,
    ca_only: bool = True,
    verbose: bool = False,
) -> List[Dict[str, float]]:
    """Runs ProteinMPNN for protein sequence design.

    This function provides an interface to ProteinMPNN, a deep learning model for protein
    sequence design. It handles the creation of temporary files for fixed positions and
    manages the execution of the ProteinMPNN command-line tool.

    Args:
        pdb_file_path (str): Path to the input PDB file.
        out_dir_root (str): Directory where designed sequences will be saved.
        all_chains (List[str]): All chains in the PDB file. Defaults to ["A"].
        pdb_path_chains (List[str]): List of chain identifiers to be designed. Defaults to ["A"].
        fix_pos (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
            Example: ["B45", "B46", "B54"]. Defaults to None.
        num_seq_per_target (int): Number of sequences to generate per target structure.
            Defaults to 8.
        omit_AAs (List[str], optional): List of amino acids that will not be considered in the design process.
            Defaults to None.
        sampling_temp (float): Temperature parameter for sequence sampling. Higher values increase
            diversity. Defaults to 0.1.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        ca_only (bool): If True, uses only alpha carbon atoms for design. Defaults to True.
        verbose (bool): If True, prints detailed output. Defaults to False.

    Returns:
        List of dictionaries, each containing:
            - 'seq': The amino acid sequence
            - 'score': The score value
            - 'seqid': The sequence recovery value

    Raises:
        ValueError: If the fix_pos format is invalid.
        RuntimeError: If ProteinMPNN command fails.
    """
    name = pdb_name_from_path(pdb_file_path)
    python_exec = os.environ.get("PYTHON_EXEC", "python")
    # Base command without optional parameters
    base_command = f"""
    {python_exec} ./ProteinMPNN/protein_mpnn_run.py \
        --pdb_path {pdb_file_path} \
        --pdb_path_chains '{' '.join(pdb_path_chains)}' \
        --out_folder {out_dir_root} \
        --num_seq_per_target {num_seq_per_target} \
        --sampling_temp {sampling_temp} \
        --omit_AAs {omit_AAs} \
        --batch_size 1 \
        --suppress_print {0 if verbose else 1} \
    """

    if ca_only:
        base_command += " --ca_only"
    if seed is not None:
        base_command += f" --seed {seed}"
    if not verbose:
        base_command += f" > /dev/null 2>&1"

    if fix_pos:
        # Initialize dictionary with sample name and empty lists for all chains
        fixed_dict = {name: {chain: [] for chain in all_chains}}

        # Fill in the fixed positions
        for pos in fix_pos:
            try:
                chain = pos[0]
                residue = int(pos[1:])
                if chain in fixed_dict[name]:
                    fixed_dict[name][chain].append(residue)
                else:
                    raise ValueError(
                        f"Chain {chain} not found in provided chain list: {all_chains}"
                    )
            except (IndexError, ValueError) as e:
                if isinstance(
                    e, ValueError
                ) and "not found in provided chain list" in str(e):
                    raise
                raise ValueError(
                    f"Invalid fix_pos format. Expected 'ChainIDNumber', got '{pos}'"
                )
        # Create fixed positions file in output directory
        fixed_positions_path = os.path.join(
            out_dir_root, f"{name}_fixed_positions.jsonl"
        )
        with open(fixed_positions_path, "w") as f:
            json.dump(fixed_dict, f)

        command = base_command + f" --fixed_positions_jsonl {fixed_positions_path}"
    else:
        command = base_command

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"ProteinMPNN command failed with error: {result.stderr}")
            raise RuntimeError(f"ProteinMPNN command failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ProteinMPNN command failed with error: {e.stderr}")
        raise RuntimeError(f"ProteinMPNN command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running ProteinMPNN: {str(e)}")
        raise RuntimeError(f"Unexpected error running ProteinMPNN: {str(e)}")

    return extract_gen_seqs(os.path.join(out_dir_root, "seqs", name + ".fa"))


## ## ## ## ## ## ## ## ## ## ## ##


def extract_seq_from_pdb(fname: str) -> str:
    """Extracts the amino acid sequence from a PDB file.

    Args:
        fname: Path to the PDB file.

    Returns:
        Single-letter amino acid sequence as a string.
    """
    protein = load_pdb(fname)
    seq = []
    for aa in protein.aatype:
        seq.append(residue_constants.restypes[aa])
    return "".join(seq)


def rmsd_metric(
    coors_1_atom37: Float[Tensor, "n 37 3"],
    coors_2_atom37: Float[Tensor, "n 37 3"],
    mask_atom_37: Optional[Bool[Tensor, "n 37"]] = None,
    mode: Literal["ca", "bb3o", "all_atom"] = "ca",
    align: bool = True,
    residue_indices: Optional[List[int]] = None,
) -> Float[Tensor, ""]:
    """
    Computes RMSD between two protein structures in the Atom37 represnetation.
    For now we only use mask to check whether we have all required atoms.

    Args:
        coors_1_atom37: First structure, shape [n, 37, 3]
        coors_2_atom37: Second structure, shape [n, 37, 3]
        mask_atom37: Binary mask of first structure, shape [n, 37]. If not provided
            defaults to all n residues present, and only allows modes "ca", "bb3o" or
            "all_atom" (see below).
        mode: Modality to use, options are
            "ca": only alpha carbon
            "bb3o": four backbone atoms (N, CA, C, O)
            "all_atom": atoms indicated by the atom37 mask
        align: Whether to align pointclouds before computing RMSD.
        residue_indices: Optional list of residue indices to compute RMSD over.
            If provided, only these residues will be included in the calculation.

    Returns:
        RMSD value, as a Torch (float) tensor with a single element
    """
    assert coors_1_atom37.shape == coors_2_atom37.shape
    assert coors_1_atom37.shape[-1] == 3
    assert coors_1_atom37.shape[-2] == 37
    assert coors_1_atom37.ndim == 3
    n = coors_1_atom37.shape[0]

    if mask_atom_37 is not None:
        assert mask_atom_37.shape == coors_1_atom37.shape[:-1]
    else:
        assert (
            mode != "all_atom"
        ), "`all_atom` mode not accepted for `rmsd_metric` when mask is not provided"
        mask_atom_37 = torch.zeros(
            (n, 37), device=coors_1_atom37.device, dtype=torch.bool
        )
        mask_atom_37[:, :3] = True  # [N CA C]
        mask_atom_37[:, 4] = True  # [O]

    # Which atoms to select, recall atom37 order [N, CA, C, CB, O, ...]
    if mode == "ca":
        mask_f = get_atom37_ca_mask(n=n, device=coors_1_atom37.device)
    elif mode == "bb3o":
        mask_f = get_atom37_bb3o_mask(n=n, device=coors_1_atom37.device)
    elif mode == "all_atom":
        mask_f = torch.ones((n, 37), device=coors_1_atom37.device, dtype=torch.bool)
    else:
        raise IOError(f"Mode {mode} for RMSD not valid")
    mask_atom_37 = mask_atom_37 * mask_f  # Keeps only requested atoms

    # If residue_indices is provided, create a residue mask and apply it
    if residue_indices is not None:
        residue_mask = torch.zeros(n, device=coors_1_atom37.device, dtype=torch.bool)
        residue_mask[residue_indices] = True
        # Apply residue mask to atom mask
        mask_atom_37 = mask_atom_37 & residue_mask.unsqueeze(1)

    coors_1 = coors_1_atom37[mask_atom_37, :]  # [num of atoms, 3]
    coors_2 = coors_2_atom37[mask_atom_37, :]  # [num of atoms, 3]

    if align:
        coors_1, coors_2 = kabsch_align_ind(coors_1, coors_2, ret_both=True)

    sq_err = (coors_1 - coors_2) ** 2
    return sq_err.sum(dim=-1).mean().sqrt().item()


def scRMSD(
    pdb_file_path: str,
    tmp_path: str = "./tmp/metrics/",
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    use_pdb_seq: bool = False,
    ret_min: bool = True,
    rmsd_modes: List[str] = ["ca"],
    motif_index: List[str] = None,
    motif_residue_indices: Optional[List[int]] = None,
    folding_models: List[Literal["esmfold", "colabfold", "chai1", "boltz2"]] = [
        "esmfold"
    ],
    # cache_dir: Optional[str] = "/lustre/fsw/portfolios/nvr/users/kdidi/.cache",
    cache_dir: Optional[str] = None,
    keep_outputs: bool = False,
) -> Dict[str, Union[float, List[float]]]:
    """
    Evaluates self-consistency RMSD metrics for given pdb.

    Args:
        pdb_file_path: Path to PDB file.
        tmp_path: Path to store files produced by ProteinMPNN and folding models.
        num_seq_per_target: Number of sequences generated by ProteinMPNN per structure.
        pmpnn_sampling_temp: ProteinMPNN sampling temperature.
        use_pdb_seq: Whether to use the sequence from the pdb file or call proteinMPNN to produce
            sequences.
        ret_min: Whether to return min RMSD or a list of all values.
        rmsd_modes: Which atoms to use to compute scRMSD, given as a list of strings. Valid
            modes are
            - "ca": alpha carbon
            - "bb3o": four backbone atoms [N CA C O]
            - "all_atom": uses the atom mask from the sequence from the PDB, this is only
            compatible with `use_pdb_seq=True`, since otherwise proteinMPNN will liekly
            return a different sequence and thus all atom comparisons cannot be done.
        motif_index (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
            Example: ["B45", "B46", "B54"]. Defaults to None.
        motif_residue_indices (List[int], optional): List of 0-indexed residue positions to compute
            RMSD over. If provided, both normal RMSD (all residues) and motif RMSD (only these residues)
            will be computed and returned.
        folding_models: List of folding models to use for structure prediction
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep the output files from the folding models after evaluation.
            If False (default), temporary directories containing PDB files are deleted to save space.
            If True, the folding model outputs are preserved for further analysis.

    Returns:
        A dictionary with results for each rmsd_mode and folding model (each one is a key). Values are
        either the best RMSD (scRMSD) or a list of all values for all generations,
        depending on the ret_min argument. When motif_residue_indices is provided, additional
        keys with "_motif" suffix are added containing motif-specific RMSD results.
    """
    for m in rmsd_modes:
        assert m in ["ca", "bb3o", "all_atom"], f"Invalid scRMSD mode {m}"
    if not use_pdb_seq:
        assert (
            "all_atom" not in rmsd_modes
        ), "`all_atom` mode not supported for scRMSD with proteinMPNN sequences"

    # Set TORCH_HOME environment variable if cache_dir is provided
    if os.getenv("CACHE_DIR"):
        cache_dir = os.getenv("CACHE_DIR")
    if cache_dir:
        os.environ["TORCH_HOME"] = cache_dir
        logger.info(f"Set TORCH_HOME to: {cache_dir}")

    name = pdb_name_from_path(pdb_file_path)
    os.makedirs(tmp_path, exist_ok=True)

    if not use_pdb_seq:
        logger.info("Running ProteinMPNN")
        gen_seqs = run_proteinmpnn(  # For now do not use keep ca_only=False
            pdb_file_path,
            tmp_path,
            num_seq_per_target=num_seq_per_target,
            sampling_temp=pmpnn_sampling_temp,
            fix_pos=motif_index,
        )  # List of sequences
        gen_seqs = [v["seq"] for v in gen_seqs]
        suffix = "mpnn"
    else:
        logger.info("Using sequence from pdb file")
        gen_seqs = [extract_seq_from_pdb(pdb_file_path)]  # List of sequences
        suffix = "pdb"

    results = {}
    for mode in rmsd_modes:
        results[mode] = {}
        # If motif evaluation is requested, also initialize motif results
        if motif_residue_indices is not None:
            results[f"{mode}_motif"] = {}

    # Load generated structure
    gen_prot = load_pdb(pdb_file_path)
    gen_coors = torch.Tensor(gen_prot.atom_positions)
    gen_mask = torch.Tensor(gen_prot.atom_mask).bool()

    # Run each folding model
    for model in folding_models:
        logger.info(f"Running {model} for {name}")

        # Create separate directory for each folding model
        model_tmp_path = os.path.join(tmp_path, f"{model}_output")
        os.makedirs(model_tmp_path, exist_ok=True)

        if model == "esmfold":
            out_folding_paths = run_esmfold(
                gen_seqs, model_tmp_path, name, suffix=suffix, cache_dir=cache_dir, keep_outputs=keep_outputs
            )
        elif model == "colabfold":
            out_folding_paths = run_colabfold(
                gen_seqs,
                model_tmp_path,
                suffix=suffix,
                cache_dir=cache_dir,
                keep_outputs=keep_outputs,
            )
        elif model == "chai1":
            out_folding_paths = run_chai1(
                gen_seqs,
                model_tmp_path,
                name,
                suffix,
                num_trunk_recycles=3,
                num_diffn_timesteps=200,
                use_esm_embeddings=False,
                device="cuda:0",
                seed=42,
                return_metrics=False,
                cache_dir=cache_dir,
                keep_outputs=keep_outputs,
            )
        elif model == "boltz2":
            out_folding_paths = run_boltz2(
                gen_seqs,
                model_tmp_path,
                name,
                suffix,
                diffusion_samples=1,
                output_format="pdb",
                return_metrics=False,
                cache_dir=cache_dir,
                keep_outputs=keep_outputs,
            )

        # print(out_folding_paths)
        
        # Compute RMSDs for each mode
        for mode in rmsd_modes:
            results[mode][model] = []
            # Initialize motif results if needed
            if motif_residue_indices is not None:
                results[f"{mode}_motif"][model] = []
            
            for out_folding in out_folding_paths:
                if out_folding is None:
                    # Handle failed predictions
                    results[mode][model].append(float("inf"))
                    if motif_residue_indices is not None:
                        results[f"{mode}_motif"][model].append(float("inf"))
                    continue

                rec_prot_folding = load_pdb(out_folding)
                rec_coors = torch.Tensor(rec_prot_folding.atom_positions)
                rec_mask = torch.Tensor(rec_prot_folding.atom_mask).bool()
                mask = gen_mask * rec_mask

                # Compute normal RMSD (all residues)
                normal_rmsd = rmsd_metric(
                    coors_1_atom37=gen_coors,
                    coors_2_atom37=rec_coors,
                    mask_atom_37=mask,
                    mode=mode,
                    residue_indices=None,  # All residues
                )
                results[mode][model].append(normal_rmsd)
                
                # Compute motif RMSD if requested
                if motif_residue_indices is not None:
                    motif_rmsd = rmsd_metric(
                        coors_1_atom37=gen_coors,
                        coors_2_atom37=rec_coors,
                        mask_atom_37=mask,
                        mode=mode,
                        residue_indices=motif_residue_indices,  # Only motif residues
                    )
                    results[f"{mode}_motif"][model].append(motif_rmsd)

        # Clean up model-specific directory to save space
        try:
            if keep_outputs:
                logger.info(f"Keeping {model} output directory: {model_tmp_path}")
            else:
                shutil.rmtree(model_tmp_path)
                logger.info(f"Cleaned up {model} output directory: {model_tmp_path}")
        except Exception as cleanup_error:
            logger.warning(
                f"Could not clean up {model} directory {model_tmp_path}: {cleanup_error}"
            )

    print(results)
    if ret_min:
        return {
            m: {model: min(results[m][model]) for model in folding_models}
            for m in results
        }
    return results


def sc_sequence_recovery(
    pdb_file_path: str,
    tmp_path: str = "./tmp/metrics/",
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    ret_max: bool = True,
    motif_index: List[str] = None,
) -> Union[float, List[float]]:
    """
    Evaluates self-consistency through sequence recovery rate for given pdb.

    Args:
        pdb_file_path: Path to PDB file.
        tmp_path: Path to store files produced by ProteinMPNN.
        num_seq_per_target: Number of sequences generated by ProteinMPNN per structure.
        pmpnn_sampling_temp: ProteinMPNN sampling temperature.
        ret_max: Whether to return max sequence recovery rate or a list of all values.
        motif_index (List[str], optional): List of positions to fix in format ["ChainID-ResidueNumber"].
        Example: ["B45", "B46", "B54"]. Defaults to None.
    Returns:
        A list of recovery rates (if ret_min is False) or a single float (the minimum).
    """

    def _rec_rate(s1: str, s2: str) -> float:
        """Calculates the sequence recovery rate between two strings.

        This function computes the fraction of positions where two sequences
        have identical characters.

        Args:
            s1: First sequence string.
            s2: Second sequence string.

        Returns:
            Recovery rate as a float between 0 and 1.

        Raises:
            ValueError: If the strings have different lengths.
        """
        if len(s1) != len(s2):
            raise ValueError("Strings must be of the same length")

        count = sum(1 for a, b in zip(s1, s2) if a == b)
        return count / len(s1)

    logger.info("Running ProteinMPNN")
    mpnn_seqs = run_proteinmpnn(  # For now do not use keep ca_only=False
        pdb_file_path,
        tmp_path,
        num_seq_per_target=num_seq_per_target,
        sampling_temp=pmpnn_sampling_temp,
        fix_pos=motif_index,
    )  # List of sequences

    pdb_seq = extract_seq_from_pdb(pdb_file_path)
    mpnn_seqs = [seq["seq"] for seq in mpnn_seqs]
    results = [_rec_rate(pdb_seq, s) for s in mpnn_seqs]
    # results_bis = [_rec_rate(pdb_seq, shuffle_string(s)) for s in mpnn_seqs]

    if ret_max:
        return max(results)
    return results
