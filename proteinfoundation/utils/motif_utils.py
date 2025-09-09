import itertools
import random
import re
from typing import List, Literal, Tuple

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from loguru import logger

from openfold.np.residue_constants import (
    atom_order,
    atom_types,
    restype_3to1,
    restype_num,
    restype_order,
)
from proteinfoundation.utils.constants import SIDECHAIN_TIP_ATOMS
from proteinfoundation.utils.coors_utils import ang_to_nm
from proteinfoundation.utils.align_utils import mean_w_mask


def _select_motif_atoms(
    available_atoms: List[int],
    atom_selection_mode: Literal[
        "ca", "bb3o", "all_atom", "tip_atoms"
    ] = "ca",
    residue_name: str = None,
) -> List[int]:
    """
    Select atoms for a residue based on the specified mode.
    
    Args:
        available_atoms: List of available atom indices for the residue
        atom_selection_mode: Mode for atom selection:
            - "ca": Only CA atoms
            - "bb3o": Backbone atoms (N, CA, C, O) 
            - "all_atom": All available atoms
            - "tip_atoms": Tip atoms of sidechains (requires residue_name)
        residue_name: Three-letter residue name (required for tip_atoms mode)
            
    Returns:
        List of selected atom indices
    """
    # Define atom indices
    backbone_atoms = [0, 1, 2, 4]  # N, CA, C, O in atom37 format
    ca_index = 1  # CA atom index in atom37 format
    
    if atom_selection_mode == "ca":
        # Select only CA atom if available
        return [ca_index] if ca_index in available_atoms else []

    elif atom_selection_mode == "bb3o":
        # Select backbone atoms (N, CA, C, O) that are available
        return [i for i in backbone_atoms if i in available_atoms]

    elif atom_selection_mode == "all_atom":
        # Select all available atoms
        return available_atoms
        
    elif atom_selection_mode == "tip_atoms":
        # Select tip atoms of sidechains based on residue type
        if residue_name is None:
            raise ValueError("residue_name must be provided for tip_atoms mode")
        
        tip_atom_names = SIDECHAIN_TIP_ATOMS.get(residue_name, [])
        tip_atom_indices = []
        for atom_name in tip_atom_names:
            if atom_name in atom_order:
                atom_idx = atom_order[atom_name]
                if atom_idx in available_atoms:
                    tip_atom_indices.append(atom_idx)
        return tip_atom_indices

    else:
        raise ValueError(f"Unknown atom selection mode: {atom_selection_mode}. Supported modes: ca, bb3o, all_atom, tip_atoms")


def generate_combinations(min_cost, max_cost, ranges):
    result = []
    ranges = [[x] if isinstance(x, int) else range(x[0], x[1] + 1) for x in ranges]
    for combination in itertools.product(*ranges):
        total_cost = sum(combination)
        if min_cost <= total_cost <= max_cost:
            padded_combination = list(combination) + [0] * (
                len(ranges) - len(combination)
            )
            result.append(padded_combination)
    return result


def generate_motif_indices(
    contig: str,
    min_length: int,
    max_length: int,
    nsamples: int = 1,
) -> Tuple[List[int], List[List[int]], List[str]]:
    """Index motif and scaffold positions by contig for sequence redesign.
    Args:
        contig (str): A string containing positions for scaffolds and motifs.

        Details:
        Scaffold parts: Contain a single integer.
        Motif parts: Start with a letter (chain ID) and contain either a single positions (e.g. A33) or a range of positions (e.g. A33-39).
        The numbers following chain IDs corresponds to the motif positions in native backbones, which are used to calculate motif reconstruction later on.
        e.g. "15/A45-65/20/A20-30"
        NOTE: The scaffold part should be DETERMINISTIC in this case as it contains information for the corresponding protein backbones.

    Raises:
        ValueError: Once a "-" is detected in scaffold parts, throws an error for the aforementioned reason.

    Returns:
        A Tuple containing:
            - overall_lengths (List[int]): Total length of the sequence defined by the contig.
            - motif_indices (List[List[int]]): List of indices where motifs are located.
            - out_strs (List[str]): String of motif indices and scaffold lengths.
    """
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    components = contig.split("/")
    ranges = []
    motif_length = 0
    for part in components:
        if part[0] in ALPHABET:
            # Motif part
            if "-" in part:
                start, end = map(int, part[1:].split("-"))
            else:
                start = end = int(part[1:])
            length = end - start + 1
            motif_length += length
        else:
            # Scaffold part
            if "-" in part:
                bounds = part.split("-")
                assert int(bounds[0]) <= int(bounds[-1])
                ranges.append((int(bounds[0]), int(bounds[-1])))
            else:
                length = int(part)
                ranges.append(length)
    combinations = generate_combinations(
        min_length - motif_length, max_length - motif_length, ranges
    )
    if len(combinations) == 0:
        raise ValueError(
            "No Motif combinations to sample from please update the max and min lengths"
        )

    overall_lengths = []
    motif_indices = []
    out_strs = []
    combos = random.choices(combinations, k=nsamples)
    for combo in combos:
        combo_idx = 0
        current_position = 1  # Start positions at 1 for 1-based indexing
        motif_index = []
        output_string = ""
        for part in components:
            if part[0] in ALPHABET:
                # Motif part
                if "-" in part:
                    start, end = map(int, part[1:].split("-"))
                else:
                    start = end = int(part[1:])
                length = end - start + 1
                motif_index.extend(range(current_position, current_position + length))
                new_part = part[0] + str(current_position)
                if length > 1:
                    new_part += "-" + str(current_position + length - 1)
                output_string += new_part + "/"
            else:
                # Scaffold part
                length = int(combo[combo_idx])
                combo_idx += 1
                output_string += str(length) + "/"
            current_position += (
                length  # Update the current position after processing each part
            )
        overall_lengths.append(current_position - 1)  # current_position is 1 past the last residue
        motif_indices.append(motif_index)
        out_strs.append(output_string[:-1])
    return (overall_lengths, motif_indices, out_strs)


def parse_motif_atom_spec(spec: str):
    """Parse a motif atom specification string into a list of (chain, res_id, [atom_names])"""
    motif_atoms = []
    for match in re.finditer(r"([A-Za-z])(\d+): \[([^\]]+)\]", spec):
        chain = match.group(1)
        res_id = int(match.group(2))
        atoms = [a.strip() for a in match.group(3).split(",")]
        motif_atoms.append((chain, res_id, atoms))
    return motif_atoms


def extract_motif_atoms_from_pdb(
    pdb_path: str,
    motif_atom_spec: str,
):
    """Efficiently extract only the specified motif atoms from a PDB using biotite."""
    array = strucio.load_structure(pdb_path, model=1)
    motif_atoms = parse_motif_atom_spec(motif_atom_spec)
    mask = np.zeros(len(array), dtype=bool)
    for chain, res_id, atom_names in motif_atoms:
        mask |= (
            (array.chain_id == chain)
            & (array.res_id == res_id)
            & np.isin(array.atom_name, atom_names)
        )
    return array[mask]


def extract_motif_from_pdb(
    position: str,
    pdb_path: str,
    motif_only: bool = False,
    motif_atom_spec: str = None,
    atom_selection_mode: Literal[
        "ca", "bb3o", "all_atom", "tip_atoms"
    ] = "ca",
    coors_to_nm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracting motif positions from input protein structure.

    Args:
        position (str): Motif region of input protein. DEMO: "A1-7/A28-79" corresponds defines res1-7 and res28-79 in chain A to be motif.
        pdb_path (str): Input protein structure, can either be a path or an AtomArray.
        motif_only (bool): Whether the pdb file only contains the motif positions.
        motif_atom_spec (str, optional): If provided, specifies motif atoms in the format "A64: [O, CG]; ...". If set, extraction is done at the atom level and only the specified atoms are returned as atom37 tensors for the corresponding residues. If not set, extraction is done at the residue/range level as before.
        atom_selection_mode (str): Mode for selecting atoms in classic mode. Options: "ca", "bb3o", "all_atom", "tip_atoms". Only used when motif_atom_spec is None.
        coors_to_nm (bool): Whether to convert motif coordinates to nanometers.

    Returns:
        motif_mask (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_motif_res, 37)
        x_motif (torch.Tensor): Motif positions in atom37 format. (n_motif_res, 37, 3)
        residue_type (torch.Tensor): Residue types of the motif. (n_motif_res)
    """
    if motif_atom_spec is not None:
        logger.info(f"Using atom-level motif specification: {motif_atom_spec[:100]}...")
        array = strucio.load_structure(pdb_path, model=1)
        motif_atoms = parse_motif_atom_spec(motif_atom_spec)
        # Get unique (chain, res_id) pairs in order
        unique_residues = []
        seen = set()
        for chain, res_id, _ in motif_atoms:
            if (chain, res_id) not in seen:
                seen.add((chain, res_id))
                unique_residues.append((chain, res_id))
        n_res = len(unique_residues)
        motif_mask = torch.zeros((n_res, 37), dtype=torch.bool)
        x_motif = torch.zeros((n_res, 37, 3), dtype=torch.float)
        residue_type = torch.ones((n_res), dtype=torch.int64) * restype_num
        for i, (chain_id, res_id) in enumerate(unique_residues):
            # Find all atom names for this residue in the motif spec
            atom_names = []
            for c, r, names in motif_atoms:
                if c == chain_id and r == res_id:
                    atom_names.extend(names)
            # Subset array for this residue
            res_mask = (array.chain_id == chain_id) & (array.res_id == res_id)
            res_atoms = array[res_mask]
            if len(res_atoms) == 0:
                continue
            res_type = restype_3to1.get(res_atoms[0].res_name, "UNK")
            residue_type[i] = restype_order.get(res_type, restype_num)
            for atom in res_atoms:
                if atom.atom_name in atom_names and atom.atom_name in atom_order:
                    atom37_idx = atom_order[atom.atom_name]
                    motif_mask[i, atom37_idx] = True
                    if coors_to_nm:
                        x_motif[i, atom37_idx] = ang_to_nm(torch.as_tensor(atom.coord))
                    else:
                        x_motif[i, atom37_idx] = torch.as_tensor(atom.coord)
        return motif_mask, x_motif, residue_type
    else:
        # Otherwise, use the old logic (residue/range based)
        position = position.split("/")
        ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        array = strucio.load_structure(pdb_path, model=1)
        motif_array = []
        seen = set()
        for i in position:
            chain_id = i[0]
            if chain_id not in ALPHABET:
                continue
            atom_mask = (array.chain_id == chain_id) & (array.hetero == False)
            if motif_only:
                if chain_id in seen:
                    continue
                else:
                    seen.add(chain_id)
            else:
                i = i.replace(chain_id, "")
                if "-" not in i:  # Single-residue motif
                    start = end = int(i)
                else:
                    start, end = i.split("-")
                    start, end = int(start), int(end)
                atom_mask = atom_mask & (array.res_id <= end) & (array.res_id >= start)
            motif_array.append(array[atom_mask])
        motif = motif_array[0]
        for i in range(len(motif_array) - 1):
            motif += motif_array[i + 1]
        # Convert motif to atom37 format
        # Get ordered unique residues by (chain_id, res_id) pairs while preserving order
        seen = set()
        unique_residues = []
        for chain, resid in zip(motif.chain_id, motif.res_id):
            if (chain, resid) not in seen:
                seen.add((chain, resid))
                unique_residues.append((chain, resid))
        n_res = len(unique_residues)

        # Initialize output arrays
        motif_mask = torch.zeros((n_res, 37), dtype=torch.bool)
        x_motif = torch.zeros((n_res, 37, 3), dtype=torch.float)
        residue_type = torch.ones((n_res), dtype=torch.int64) * restype_num

        # Map each residue's atoms to atom37 format
        for i, (chain_id, res_id) in enumerate(unique_residues):
            # Get atoms for this specific residue
            res_mask = (motif.chain_id == chain_id) & (motif.res_id == res_id)
            res_atoms = motif[res_mask]
            res_type = restype_3to1.get(res_atoms[0].res_name, "UNK")
            residue_type[i] = restype_order.get(res_type, restype_num)

            # Get available atom indices for this residue
            available_atom_indices = []
            for atom in res_atoms:
                if atom.atom_name in atom_order:
                    atom37_idx = atom_order[atom.atom_name]
                    available_atom_indices.append(atom37_idx)

            # Select atoms based on the specified mode
            if len(available_atom_indices) > 0:
                selected_atom_indices = _select_motif_atoms(
                    available_atom_indices, atom_selection_mode, res_atoms[0].res_name
                )

                # Map selected atoms to their positions in atom37 format
                for atom in res_atoms:
                    if atom.atom_name in atom_order:
                        atom37_idx = atom_order[atom.atom_name]
                        if atom37_idx in selected_atom_indices:
                            motif_mask[i, atom37_idx] = True
                            if coors_to_nm:
                                x_motif[i, atom37_idx] = ang_to_nm(torch.as_tensor(atom.coord))
                            else:
                                x_motif[i, atom37_idx] = torch.as_tensor(atom.coord)
        # center motif
        motif_center = mean_w_mask(x_motif.flatten(0, 1), motif_mask.flatten(0, 1)).unsqueeze(0)
        x_motif = x_motif - motif_center
        x_motif = x_motif * motif_mask[..., None]  # Is this needed?
        return motif_mask, x_motif, residue_type


def pad_motif_to_full_length(
    motif_mask: torch.Tensor,
    x_motif: torch.Tensor,
    residue_type: torch.Tensor,
    contig_string: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad motif to full length.
    Args:
        motif_mask (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_motif_res, 37)
        x_motif (torch.Tensor): Motif positions in atom37 format. (n_motif_res, 37, 3)
        residue_type (torch.Tensor): Residue types of the motif. (n_motif_res)
        contig_string (str): Contig string containing motif positions.

    Returns:
        motif_mask_full (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_full_length, 37)
        x_motif_full (torch.Tensor): Motif positions in atom37 format. (n_full_length, 37, 3)
        residue_type_full (torch.Tensor): Residue types of the motif. (n_full_length)
    """
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    components = contig_string.split("/")
    current_position = 1  # Start positions at 1 for 1-based indexing
    motif_index = []
    for part in components:
        if part[0] in ALPHABET:
            # Motif part
            if "-" in part:
                start, end = map(int, part[1:].split("-"))
            else:
                start = end = int(part[1:])
            length = end - start + 1
            motif_index.extend(range(current_position, current_position + length))
        else:
            # Scaffold part
            length = int(part)
        current_position += (
            length  # Update the current position after processing each part
        )

    # current_position is 1 past the last residue, so subtract 1 for actual length
    actual_length = current_position - 1
    motif_index = (
        torch.tensor(motif_index, dtype=torch.int64) - 1
    )  # Change to 0-based indexing
    motif_mask_full = torch.zeros((actual_length, 37), dtype=torch.bool)
    x_motif_full = torch.zeros((actual_length, 37, 3), dtype=torch.float)
    residue_type_full = torch.ones((actual_length,), dtype=torch.int64) * restype_num
    motif_mask_full[motif_index] = motif_mask
    x_motif_full[motif_index] = x_motif
    residue_type_full[motif_index] = residue_type
    return motif_mask_full, x_motif_full, residue_type_full


def pad_motif_to_full_length_unindexed(
    motif_mask: torch.Tensor,
    x_motif: torch.Tensor,
    residue_type: torch.Tensor,
    gen_coors: torch.Tensor,
    gen_mask: torch.Tensor,
    gen_aa_type: torch.Tensor,
    match_aatype: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds indices greedily matching each motif residue to the closest residue in the generated protein.
    Also accounts for residue type, which should match, if match_aatype is True.

    When running with match_aatype=True, the function will try to match the residue types of the motif and the generated protein.
    If no match is found, the function will try again with match_aatype=False.

    Since a success involves generating the right residue type, unning this function with match_aatype=True strictly
    increases the likelihood of success.

    If no match is found without accounting for residue types, the function defaults to matching the motif to the first n_motif residues.

    Args:
        motif_mask (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_motif_res, 37)
        x_motif (torch.Tensor): Motif positions in atom37 format. (n_motif_res, 37, 3)
        residue_type (torch.Tensor): Residue types of the motif. (n_motif_res)
        gen_coors (torch.Tensor): Coordinates of the generated protein. (n_res, 37, 3)
        gen_mask (torch.Tensor): Mask of the generated protein. (n_res, 37)
        gen_aa_type (torch.Tensor): Residue types of the generated protein. (n_res)
        match_aatype (bool): Whether to match residue types. Leave as True.

    Returns:
        motif_mask_full (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_full_length, 37)
        x_motif_full (torch.Tensor): Motif positions in atom37 format. (n_full_length, 37, 3)
        residue_type_full (torch.Tensor): Residue types of the motif. (n_full_length)
    """
    nres = gen_coors.shape[0]
    nres_motif = x_motif.shape[0]
    motif_index = []

    # Greedily match each motif component to the best component of the generated protein
    # no aligning individual comparisons, since that could break the rigid constraint in the motif
    for i in range(nres_motif):
        # Define motif i-th component
        motif_mask_i = motif_mask[i]  # (37)
        x_motif_i = x_motif[i]  # (37, 3)
        aatype_motif_i = residue_type[i]  # int
        
        # Find the best match for the i-th motif component
        best_match_idx = None
        best_rmsd = float('inf')

        for j in range(nres):
            # Define generated j-th component
            gen_mask_j = gen_mask[j]  # (37)
            gen_coors_j = gen_coors[j]  # (37, 3)
            aatype_gen_j = gen_aa_type[j]  # int

            mask_motif_i_gen_j = motif_mask_i & gen_mask_j  # [37]
            
            if mask_motif_i_gen_j.sum() == 0:
                # No overlap in atoms
                continue

            x_motif_i_subset = x_motif_i[mask_motif_i_gen_j]  # [nres_mi_gj, 3]
            gen_coors_j_subset = gen_coors_j[mask_motif_i_gen_j]  # [nres_mi_gj, 3]

            rmsd = torch.sqrt(torch.sum((x_motif_i_subset - gen_coors_j_subset) ** 2, dim=1).mean())

            # Get improvement condition
            cond = rmsd < best_rmsd and j not in motif_index
            if match_aatype:
                cond = cond and aatype_motif_i == aatype_gen_j
            
            # Update best match if improved
            if cond:
                best_rmsd = rmsd
                best_match_idx = j

        if best_match_idx is None:
            logger.warning(f"No best match found for motif component {i} with match_aatype={match_aatype}")

        motif_index.append(best_match_idx)

    if None in motif_index:
        # There was some issue in the matching process, defaults to the first n residues
        motif_index = [i for i in range(nres_motif)]
        logger.warning("\n\n\nError during matching, defaulting to the first n residues\n\n\n")

    motif_mask_full = torch.zeros((nres, 37), dtype=torch.bool)
    x_motif_full = torch.zeros((nres, 37, 3), dtype=torch.float)
    residue_type_full = torch.ones((nres,), dtype=torch.int64) * restype_num
    motif_mask_full[motif_index] = motif_mask
    x_motif_full[motif_index] = x_motif
    residue_type_full[motif_index] = residue_type
    return motif_mask_full, x_motif_full, residue_type_full


def parse_motif(
    motif_pdb_path: str,
    contig_string: str = None,
    nsamples: int = 1,
    motif_only: bool = False,
    motif_min_length: int = None,
    motif_max_length: int = None,
    segment_order: str = None,
    motif_atom_spec: str = None,
    atom_selection_mode: Literal[
        "ca", "bb3o", "all_atom", "tip_atoms"
    ] = "ca",
) -> Tuple[
    List[int], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[str]
]:
    """
    Extract motif positions from input protein structure and generate motif indices and mask.

    This function supports two modes of motif specification:

    1. **Atom-level specification** (when motif_atom_spec is provided):
       - Allows precise specification of which atoms to include for each residue
       - Format: "A64: [O, CG]; B12: [N, CA]; ..."
       - atom_selection_mode is ignored in this mode

    2. **Residue/range-based specification** (when motif_atom_spec is None):
       - Uses contig_string to specify residue ranges (e.g., "A1-7/A28-79")
       - atom_selection_mode determines which atoms are selected for each residue
       - Options: "ca", "bb3o", "all_atom", "tip_atoms"

    Args:
        motif_pdb_path (str): Path to the input protein structure.
        contig_string (str): Contig string containing motif positions (used in mode 2).
        nsamples (int): Number of samples to generate.
        motif_only (bool): Whether to extract only motif positions.
        motif_min_length (int): Minimum length of the motif.
        motif_max_length (int): Maximum length of the motif.
        segment_order (str): Optional segment order.
        motif_atom_spec (str, optional): Atom-level specification (mode 1).
            Format: "A64: [O, CG]; B12: [N, CA]; ..." If provided, uses atom-level extraction.
        atom_selection_mode (str): Atom selection mode for residue/range-based extraction (mode 2).
            Options:
            - "ca": Select only CA atoms (default, fastest)
            - "bb3o": Select backbone atoms (N, CA, C, O)
            - "all_atom": Select all available atoms (most complete)
            - "tip_atoms": Select tip atoms of sidechains

    Returns:
        lengths (List[int]): List of motif lengths.
        motif_masks (List[torch.Tensor]): List of full motif masks.  (n_res, 37)
        x_motifs (List[torch.Tensor]): List of full motif positions. (n_res, 37, 3)
        residue_types (List[torch.Tensor]): List of full motif residue types. (n_res)
        out_strs (List[str] or None): List of motif indices and scaffold lengths (None for atom-level extraction).

    Example:
        # Mode 1: Atom-level specification
        parse_motif(
            motif_pdb_path="motif.pdb",
            motif_atom_spec="A64: [O, CG]; A65: [N, CA]",
            # atom_selection_mode is ignored
        )

        # Mode 2: Residue/range-based with different atom selection modes
        parse_motif(
            motif_pdb_path="motif.pdb",
            contig_string="A1-7/A28-79",
            atom_selection_mode="tip_atoms"  # or "ca", "bb3o", "all_atom", etc.
        )
    """
    if motif_atom_spec is not None:
        logger.info(f"Using atom-level motif specification: {motif_atom_spec[:100]}...")
        motif_mask, x_motif, residue_type = extract_motif_from_pdb(
            None, motif_pdb_path, motif_atom_spec=motif_atom_spec
        )
        n_res = motif_mask.shape[0]
        # For consistency, wrap in lists as in the old code
        return [n_res], [motif_mask], [x_motif], [residue_type], [None]

    # Validate atom_selection_mode for classic mode
    valid_modes = ["ca", "bb3o", "all_atom", "tip_atoms"]
    if atom_selection_mode not in valid_modes:
        raise ValueError(
            f"Invalid atom_selection_mode '{atom_selection_mode}'. "
            f"Must be one of: {valid_modes}"
        )

    logger.info(
        f"Using residue/range-based motif specification with atom_selection_mode='{atom_selection_mode}'"
    )
    if contig_string:
        logger.info(f"Contig string: {contig_string}")

    # Otherwise, use the old logic
    motif_mask, x_motif, residue_type = extract_motif_from_pdb(
        contig_string,
        motif_pdb_path,
        motif_only=motif_only,
        atom_selection_mode=atom_selection_mode,
    )

    lengths, motif_indices, out_strs = generate_motif_indices(
        contig_string, motif_min_length, motif_max_length, nsamples
    )
    motif_masks = []
    x_motifs = []
    residue_types = []
    # print(lengths)
    # print(motif_indices)
    # print(out_strs)
    for length, motif_index, _ in zip(lengths, motif_indices, out_strs):
        # Construct motif_mask
        cur_mask = torch.zeros((length, 37), dtype=torch.bool)
        assert (
            len(motif_index) == motif_mask.shape[0] == x_motif.shape[0]
        ), f"motif_index: {len(motif_index)}, motif_mask: {motif_mask.shape[0]}, x_motif: {x_motif.shape[0]}, lengths don't match"
        motif_index = (
            torch.tensor(motif_index, dtype=torch.int64) - 1
        )  # Change to 0-based indexing
        cur_mask[motif_index] = motif_mask

        # Construct full structure with zero padding for the scaffold
        cur_motif = torch.zeros((length, 37, 3), dtype=x_motif.dtype)
        cur_motif[motif_index] = x_motif
        cur_residue_type = torch.ones((length), dtype=torch.int64) * restype_num
        cur_residue_type[motif_index] = residue_type
        motif_masks.append(cur_mask)
        x_motifs.append(cur_motif)
        residue_types.append(cur_residue_type)
    # print([x.shape for x in motif_masks])
    # exit()
    #! this is already in nanometers 
    return lengths, motif_masks, x_motifs, residue_types, out_strs


def save_motif_csv(pdb_path, motif_task_name, contigs, outpath=None, segment_order="A"):
    pdb_name = pdb_path.split("/")[-1].split(".")[0]

    # Create a list of dictionaries to be converted into a DataFrame
    # Each dictionary represents a row in the CSV file
    data = [
        {
            "pdb_name": pdb_name,
            "sample_num": index,
            "contig": value,
            "redesign_positions": " ",  #';'.join([x for x in value.split('/') if 'A' in x or 'B' in x or 'C' in x or 'D' in x]),
            "segment_order": segment_order,
        }
        for index, value in enumerate(contigs)
    ]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    if outpath is None:
        outpath = f"./{motif_task_name}_motif_info.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(outpath, index=False)
