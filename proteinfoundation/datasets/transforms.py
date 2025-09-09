import random
from typing import List, Literal
import numpy as np
import torch
from torch_geometric import transforms as T
from torch_geometric.data import Data

from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils
from proteinfoundation.utils.align_utils import mean_w_mask
from proteinfoundation.utils.constants import SIDECHAIN_TIP_ATOMS
from proteinfoundation.utils.coors_utils import ang_to_nm, sample_uniform_rotation


class CopyCoordinatesTransform(T.BaseTransform):
    """Copies coords to coords_unmodified. Useful if other transforms like noising or rotations/translations are applied later on."""

    def __call__(self, graph: Data) -> Data:
        graph.coords_unmodified = graph.coords.clone()


class ChainBreakCountingTransform(T.BaseTransform):
    """Counting the number of chain breaks in the protein coordinates and saving it as an attribute."""

    def __init__(
        self,
        chain_break_cutoff: float = 4.0,
    ):
        self.chain_break_cutoff = chain_break_cutoff

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        graph.chain_breaks = (ca_dists > self.chain_break_cutoff).sum().item()
        return graph


class ChainBreakPerResidueTransform(T.BaseTransform):
    """Creates a binary mask indicating whether residue has chain break or not."""

    def __init__(
        self,
        chain_break_cutoff: float = 4.0,
    ):
        self.chain_break_cutoff = chain_break_cutoff

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        chain_breaks_per_residue = ca_dists > self.chain_break_cutoff
        graph.chain_breaks_per_residue = torch.cat(
            (
                chain_breaks_per_residue,
                torch.tensor(
                    [False], dtype=torch.bool, device=chain_breaks_per_residue.device
                ),
            )
        )
        return graph


class PaddingTransform(T.BaseTransform):
    def __init__(self, max_size=256, fill_value=0):
        self.max_size = max_size
        self.fill_value = fill_value

    def __call__(self, graph: Data) -> Data:
        for key, value in graph:
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1:  # Only pad tensors with 2 or more dimensions
                    pad_dim = 0
                    graph[key] = self.pad_tensor(
                        value, self.max_size, pad_dim, self.fill_value
                    )
        return graph

    def pad_tensor(self, tensor, max_size, dim, fill_value=0):
        if tensor.size(dim) >= max_size:
            return tensor

        pad_size = max_size - tensor.size(dim)
        padding = [0] * (2 * tensor.dim())
        padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
        return torch.nn.functional.pad(
            tensor, pad=tuple(padding), mode="constant", value=fill_value
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_size={self.max_size}, fill_value={self.fill_value})"


class GlobalRotationTransform(T.BaseTransform):
    """Modifies the global rotation of the atom37 representation randomly.

    Should be used as the first transform in the pipeline that modifies coordinates in order to keep
    e.g. frame construction or other things consistent down the pipeline."""

    def __init__(self, rotation_strategy: Literal["uniform"] = "uniform"):
        self.rotation_strategy = rotation_strategy

    def __call__(self, graph: Data) -> Data:
        if self.rotation_strategy == "uniform":
            rot = sample_uniform_rotation(
                dtype=graph.coords_nm.dtype, device=graph.coords_nm.device
            )
        else:
            raise ValueError(
                f"Rotation strategy {self.rotation_strategy} not supported"
            )
        graph.coords_nm = torch.matmul(
            graph.coords_nm, rot
        )  # [n, 37, 3] * [3, 3] -> [n, 37, 3]
        return graph


class StructureNoiseTransform(T.BaseTransform):
    """Adds noise to the coordinates of a protein structure.

    Sets the following attributes on the protein data object:
        coords_uncorrupted (torch.Tensor): The original coordinates of the protein.
        noise (torch.Tensor): The noise added to the coordinates.
        coords (torch.Tensor): The original coordinates with added noise.

    Args:
        corruption_rate (float): Magnitude of corruption to apply to the coordinates.
        corruption_strategy (str): Noise strategy to use for corruption. Must be
            either "uniform" or "gaussian".
        gaussian_mean (float, optional): Mean of the Gaussian distribution.
            Defaults to 0.0.
        gaussian_std (float, optional): Standard deviation of the Gaussian
            distribution. Defaults to 1.0.
        uniform_min (float, optional): Minimum value of the uniform distribution.
            Defaults to -1.0.
        uniform_max (float, optional): Maximum value of the uniform distribution.
            Defaults to 1.0.

    Raises:
        ValueError: If the corruption strategy is not supported.
    """

    def __init__(
        self,
        corruption_strategy: Literal["uniform", "gaussian"] = "gaussian",
        gaussian_mean: float = 0.0,
        gaussian_std: float = 1.0,
        uniform_min: float = -1.0,
        uniform_max: float = 1.0,
    ):
        self.corruption_strategy = corruption_strategy
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max

    def __call__(self, graph: Data) -> Data:
        """Adds noise to the coordinates of a protein structure.

        Args:
            graph (torch_geometric.data.Data): Protein data object.

        Returns:
            torch_geometric.data.Data: Protein data object with corrupted coordinates.
        """

        if self.corruption_strategy == "uniform":
            noise = torch.empty_like(graph.coords_nm).uniform_(
                self.uniform_min, self.uniform_max
            )
        elif self.corruption_strategy == "gaussian":
            noise = torch.normal(
                mean=self.gaussian_mean,
                std=self.gaussian_std,
                size=graph.coords_nm.size(),
            )
        else:
            raise ValueError(
                f"Corruption strategy '{self.corruption_strategy}' not supported."
            )

        graph.noise = noise
        graph.noise[graph.coord_mask == 0] = 0
        graph.coords_nm += noise
        return graph

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(corruption_strategy={self.corruption_strategy}, "
            f"gaussian_mean={self.gaussian_mean}, "
            f"gaussian_std={self.gaussian_std}, uniform_min={self.uniform_min}, "
            f"uniform_max={self.uniform_max})"
        )


class CenterStructureTransform(T.BaseTransform):
    """Centers the structure based on CA coordinates."""

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords_nm[:, 1, :]  # [n, 3]
        mask = torch.ones(ca_coords.shape[0], dtype=torch.bool, device=ca_coords.device)
        com = mean_w_mask(ca_coords, mask, keepdim=True)  # [1, 3]
        graph.coords_nm = graph.coords_nm - com[None, ...]  # [n, 37, 3] - [1, 3]
        return graph


class GlobalTranslationTransform(T.BaseTransform):
    """Applies a global translation to the coordinates of a protein structure."""

    def __init__(
        self,
        translation_strategy: Literal["uniform", "normal"] = "uniform",
        uniform_min: float = -1.0,
        uniform_max: float = 1.0,
        normal_mean: float = 0.0,
        normal_std: float = 1.0,
    ):
        self.translation_strategy = translation_strategy
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max
        self.normal_mean = normal_mean
        self.normal_std = normal_std

    def __call__(self, graph: Data) -> Data:

        if self.translation_strategy == "uniform":
            translation = torch.empty(
                3, dtype=graph.coords_nm.dtype, device=graph.coords_nm.device
            ).uniform_(self.uniform_min, self.uniform_max)
        elif self.translation_strategy == "normal":
            translation = torch.normal(
                mean=self.normal_mean,
                std=self.normal_std,
                size=(3,),
                dtype=graph.coords_nm.dtype,
                device=graph.coords_nm.device,
            )
        else:
            raise ValueError(
                f"Translation strategy '{self.translation_strategy}' not supported."
            )

        graph.translation = translation
        graph.coords_nm += translation
        return graph

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(translation_strategy={self.translation_strategy}, "
            f"gaussian_mean={self.normal_mean}, gaussian_std={self.normal_std}, "
            f"uniform_min={self.uniform_min}, uniform_max={self.uniform_max})"
        )


class CoordsToNanometers(T.BaseTransform):
    """Gets cordinates in nanometers."""

    def __call__(self, graph: Data) -> Data:
        graph.coords_nm = ang_to_nm(graph.coords)
        return graph


class OpenFoldFrameTransform(T.BaseTransform):
    """OpenFold frame transform."""

    def __call__(self, graph: Data) -> Data:
        aatype = torch.zeros_like(graph.residue_type).long()
        coords = graph.coords.double()
        atom_mask = graph.coord_mask.double()
        # Run through OpenFold data transforms.
        chain_feats = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": atom_mask,
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats["rigidgroups_gt_frames"]
        )[:, 0]
        rotations_gt = rigids_1.get_rots().get_rot_mats()
        translations_gt = rigids_1.get_trans()

        graph.translations_gt = translations_gt
        graph.rotations_gt = rotations_gt

        return graph


class CenteringTransform(T.BaseTransform):
    """Centers protein structures based on one of their chains or a provided mask."""

    def __init__(
        self,
        center_mode: str = "full",
        data_mode: str = "bb_ca",
        variance_perturbation: float = 0.01,
    ) -> None:
        """Initializes the transform with the chain ID to center on.
        Args:
            center_mode (str): type of centering to perform. Options:
                - "full": Center on all atoms
                - "random_chain": Center on a random chain
                - "random_unique_chain": Center on a random unique chain
                - "motif": Center on motif mask if available
                - "stochastic_centering": center whole protein, then add stochastic translation
            data_mode (str): The data to center on. Options:
                - "bb_ca": Center on CA atoms only
                - "all-atom": Center on all atoms
            variance_perturbation (float): Variance of the stochastic translation if enabled. Defaults to 0.01.
        """
        self.center_mode = center_mode
        self.data_mode = data_mode
        self.variance_perturbation = variance_perturbation

    def __call__(self, graph: Data) -> Data:
        """Centers the graph based on the center mode and data mode.
        Args:
            graph (Data): The graph to center

        Returns:
            Data: The centered graph
        """
        # set the correct mask for centering depending on task
        if self.center_mode == "full" or self.center_mode == "stochastic_centering":
            centering_mask = torch.ones_like(
                graph["chains"], dtype=torch.bool
            )  # no target specified
        elif self.center_mode == "random_chain":
            # choose a random chain
            random_chain = np.random.choice(graph.chains[-1].item() + 1)
            centering_mask = graph["chains"] == random_chain
        elif self.center_mode == "random_unique_chain":
            # choose a random unique chain
            names, counts = np.unique(graph.chain_names, return_counts=True)
            unique_names = names[counts == 1]
            if len(unique_names) == 0:
                raise ValueError(f"No unique chain found in pdb: {graph.id}")
            random_name = np.random.choice(unique_names)
            random_chain = graph.chain_names.index(random_name)
            centering_mask = graph["chains"] == random_chain
        elif self.center_mode == "motif":
            if not hasattr(graph, "motif_mask"):
                raise ValueError(
                    "Motif mask not found in graph. Apply MotifMaskTransform first."
                )
            centering_mask = graph.motif_mask.flatten(0, 1)
        else:
            raise ValueError(f"Invalid center mode {self.center_mode}")

        # set the correct data mode for centering
        if self.data_mode == "bb_ca":
            coords = graph.coords_nm[:, 1, :]
        elif self.data_mode == "all-atom":
            coords = graph.coords_nm.flatten(0, 1)
        else:
            raise ValueError(f"Invalid data mode {self.data_mode}")

        # get the mean of the selected chain
        masked_mean = mean_w_mask(coords, centering_mask, keepdim=True)
        # If stochastic centering, add random translation to masked mean that is used for centering
        if self.center_mode == "stochastic_centering":
            translation = torch.normal(
                mean=0.0,
                std=self.variance_perturbation**0.5,
                size=(3,),
                dtype=graph.coords.dtype,
                device=graph.coords.device,
            )
            masked_mean += translation
            graph.stochastic_translation = translation
        # substract the mean of that chain from all coordinates
        if self.data_mode == "bb_ca":
            graph["coords_nm"] -= masked_mean
        else:  # all-atom
            nres = graph.coords_nm.shape[0]
            graph["coords_nm"] = graph["coords_nm"].flatten(0, 1) - masked_mean
            graph["coords_nm"] = graph["coords_nm"].view(
                nres, -1, 3
            )
        return graph


class MotifMaskTransform(T.BaseTransform):
    """
    Creates a motif mask for a protein structure, supporting multiple residue and atom selection strategies.

    Args:
        atom_selection_mode (str): How to select atoms within each motif residue. Options:
            - "random": Randomly select between 1 and all available atoms per residue.
            - "backbone": Select only backbone atoms (N, CA, C, O).
            - "sidechain": Select only sidechain atoms (all non-backbone atoms).
            - "all": Select all available atoms.
            - "ca_only": Select only CA atom if available.
            - "tip_atoms": Select only the tip atoms of sidechains (e.g., OH for Ser, NH2 for Arg).
            - "bond_graph": Use a chemically-aware expansion from a seed atom using the residue's bond graph.
        residue_selection_mode (str): How to select which residues are in the motif. Options:
            - "relative_fraction": Select a fraction of residues (see motif_min_pct_res, motif_max_pct_res, and segment logic applies).
            - "absolute_number": Select a fixed number of residues (see motif_min_n_res, motif_max_n_res, each residue is its own segment).
        motif_prob (float, optional): Probability of creating a motif. Defaults to 1.0.
        # Used if residue_selection_mode == "relative_fraction":
        motif_min_pct_res (float, optional): Minimum percentage of residues in motif. Defaults to 0.05.
        motif_max_pct_res (float, optional): Maximum percentage of residues in motif. Defaults to 0.5.
        motif_min_n_seg (int, optional): Minimum number of segments in motif. Defaults to 1.
        motif_max_n_seg (int, optional): Maximum number of segments in motif. Defaults to 4.
        # Used if residue_selection_mode == "absolute_number":
        motif_min_n_res (int, optional): Minimum number of residues in motif. Defaults to 1.
        motif_max_n_res (int, optional): Maximum number of residues in motif. Defaults to 8.

    Returns:
        The input graph with the following attributes added:
            - motif_mask: Binary tensor of shape (num_res, 37) indicating which atoms are in the motif.
            - x_motif: Tensor containing fixed coordinates for the motif.
            - seq_motif_mask: Binary tensor of shape (num_res) indicating which residues are in the motif.
            - seq_motif: Tensor containing the motif sequence.

    Notes:
        - For 'bond_graph' atom selection, the motif is expanded from a seed atom using the residue's bond graph (from AlphaFold's stereo_chemical_props).
        - In 'absolute_number' mode, each selected residue is its own segment and residues are chosen randomly (not necessarily contiguous).
        - In 'relative_fraction' mode, the selected residues are grouped into contiguous segments.
    """

    def __init__(
        self,
        atom_selection_mode: Literal[
            "random",
            "backbone",
            "sidechain",
            "all",
            "ca_only",
            "tip_atoms",
            "bond_graph",
        ] = "ca_only",
        residue_selection_mode: Literal[
            "relative_fraction", "absolute_number"
        ] = "relative_fraction",
        motif_prob: float = 1.0,
        motif_min_pct_res: float = 0.05,
        motif_max_pct_res: float = 0.5,
        motif_min_n_seg: int = 1,
        motif_max_n_seg: int = 4,
        motif_min_n_res: int = 1,
        motif_max_n_res: int = 8,
    ):
        self.atom_selection_mode = atom_selection_mode
        self.residue_selection_mode = residue_selection_mode
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg
        self.motif_min_n_res = motif_min_n_res
        self.motif_max_n_res = motif_max_n_res

        # Define backbone atom indices based on atom_types from residue_constants
        self.backbone_atoms = [
            residue_constants.atom_types.index("N"),
            residue_constants.atom_types.index("CA"),
            residue_constants.atom_types.index("C"),
            residue_constants.atom_types.index("O"),
        ]
        self.ca_index = residue_constants.atom_types.index("CA")

    def _select_atoms(
        self, available_atoms: torch.Tensor, residue_idx: int = None, graph: Data = None
    ) -> List[int]:
        """Select atoms for a residue based on the specified mode.

        Args:
            available_atoms (torch.Tensor): Tensor of available atom indices.
            residue_idx (int, optional): Residue index in the graph (needed for bond_graph mode).
            graph (Data, optional): The full graph (needed for bond_graph mode).

        Returns:
            List[int]: List of selected atom indices.
        """
        if self.atom_selection_mode == "random":
            n_atoms = random.randint(1, len(available_atoms))
            return random.sample(available_atoms.tolist(), n_atoms)

        elif self.atom_selection_mode == "backbone":
            # Select only backbone atoms that are available
            return [i for i in self.backbone_atoms if i in available_atoms]

        elif self.atom_selection_mode == "sidechain":
            # Select only sidechain atoms (all non-backbone atoms)
            sidechain_atoms = [
                i for i in available_atoms if i not in self.backbone_atoms
            ]
            if len(sidechain_atoms) > 0:
                n_atoms = random.randint(1, len(sidechain_atoms))
                return random.sample(sidechain_atoms, n_atoms)
            return []

        elif self.atom_selection_mode == "all":
            # Select all available atoms
            return available_atoms.tolist()

        elif self.atom_selection_mode == "ca_only":
            # Select only CA atom if available
            return [self.ca_index] if self.ca_index in available_atoms else []

        elif self.atom_selection_mode == "tip_atoms":
            # Select only tip atoms of sidechains
            tip_atoms = [i for i in available_atoms if i in SIDECHAIN_TIP_ATOMS]
            return tip_atoms

        elif self.atom_selection_mode == "bond_graph":
            if graph is None or residue_idx is None:
                raise ValueError(
                    "graph and residue_idx must be provided for bond_graph mode"
                )
            # Always use backbone O atom index for this residue
            ref_atom_idx = residue_constants.atom_order["O"]
            if ref_atom_idx not in available_atoms:
                ref_atom_idx = residue_constants.atom_order["CA"]
            ref_atom_coord = graph.coords_nm[residue_idx, ref_atom_idx, :]
            atom_coords = graph.coords_nm[residue_idx, available_atoms, :]
            dists = torch.norm(atom_coords - ref_atom_coord, dim=-1)
            # 80%: farthest from ref_atom, 20%: random
            if random.random() < 0.8:
                seed_atom_idx = torch.argmax(dists).item()
            else:
                seed_atom_idx = random.randint(0, len(available_atoms) - 1)
            seed_atom = available_atoms[seed_atom_idx].item()
            # Build bond graph using residue_constants
            res_type_idx = graph.residue_type[residue_idx].item()
            resname = residue_constants.restype_1to3.get(
                residue_constants.restypes[res_type_idx], "UNK"
            )
            residue_bonds, _, _ = residue_constants.load_stereo_chemical_props()
            bonds = residue_bonds.get(resname, [])
            # Map atom names to local indices in available_atoms
            atom_name_to_local_idx = {
                residue_constants.atom_types[atom_idx]: i
                for i, atom_idx in enumerate(available_atoms.tolist())
                if atom_idx < len(residue_constants.atom_types)
            }
            n_atoms = len(available_atoms)
            adj = torch.zeros((n_atoms, n_atoms), dtype=torch.bool)
            for bond in bonds:
                a1 = bond.atom1_name
                a2 = bond.atom2_name
                if a1 in atom_name_to_local_idx and a2 in atom_name_to_local_idx:
                    i = atom_name_to_local_idx[a1]
                    j = atom_name_to_local_idx[a2]
                    adj[i, j] = True
                    adj[j, i] = True
            # If no bonds found, fallback to fully connected
            if adj.sum() == 0:
                adj = torch.ones((n_atoms, n_atoms), dtype=torch.bool)
                torch.diagonal(adj).fill_(0)
            # Map atom indices to local indices
            atom_idx_map = {atom.item(): i for i, atom in enumerate(available_atoms)}
            # BFS expansion from seed_atom
            n_expand = np.random.geometric(p=0.5)
            visited = set([seed_atom])
            queue = [seed_atom]
            while len(visited) < min(n_expand, len(available_atoms)) and queue:
                current = queue.pop(0)
                current_local = atom_idx_map[current]
                neighbors = [
                    available_atoms[i].item()
                    for i in range(n_atoms)
                    if adj[current_local, i]
                ]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    if len(visited) >= min(n_expand, len(available_atoms)):
                        break
            return list(visited)
        else:
            raise ValueError(f"Unknown atom selection mode: {self.atom_selection_mode}")

    def __call__(self, graph: Data) -> Data:
        if random.random() > self.motif_prob:
            motif_mask = torch.zeros_like(graph.coord_mask)
            graph.motif_mask = motif_mask.bool()
            return graph

        num_res = graph.coords_nm.shape[0]
        if self.residue_selection_mode == "relative_fraction":
            motif_n_res = int(
                random.random()
                * (num_res * self.motif_max_pct_res - num_res * self.motif_min_pct_res)
                + num_res * self.motif_min_pct_res
            )
            motif_n_res = min(motif_n_res, num_res)
            # Segment logic: contiguous segments within the selected motif region
            motif_n_seg = int(
                random.random()
                * (min(motif_n_res, self.motif_max_n_seg) - self.motif_min_n_seg + 1)
                + self.motif_min_n_seg
            )
            indices = (
                np.sort(random.sample(range(1, motif_n_res), motif_n_seg - 1))
                if motif_n_seg > 1
                else []
            )
            indices = (
                np.concatenate([[0], indices, [motif_n_res]]).astype(int)
                if motif_n_seg > 1
                else np.array([0, motif_n_res])
            )
            segments = []
            for i in range(len(indices) - 1):
                start, end = indices[i], indices[i + 1]
                segment_length = end - start
                segments.append("".join(["1"] * segment_length))
            segments.extend(["0"] * (num_res - motif_n_res))
            random.shuffle(segments)
            motif_sequence_mask = torch.tensor(
                [int(elt) for elt in "".join(segments)]
            ).bool()
        elif self.residue_selection_mode == "absolute_number":
            motif_n_res = random.randint(self.motif_min_n_res, self.motif_max_n_res)
            motif_n_res = min(motif_n_res, num_res)
            # Randomly pick motif_n_res unique residue indices
            motif_indices = random.sample(range(num_res), motif_n_res)
            motif_sequence_mask = torch.zeros(num_res, dtype=torch.bool)
            motif_sequence_mask[motif_indices] = True
        else:
            raise ValueError(
                f"Unknown residue_selection_mode: {self.residue_selection_mode}"
            )
        motif_mask = torch.zeros_like(graph.coord_mask)
        for res_idx in torch.where(motif_sequence_mask)[0]:
            available_atoms = torch.where(graph.coord_mask[res_idx])[0]
            if len(available_atoms) == 0:
                continue
            if self.atom_selection_mode == "bond_graph":
                selected_atoms = self._select_atoms(
                    available_atoms, residue_idx=res_idx.item(), graph=graph
                )
            else:
                selected_atoms = self._select_atoms(available_atoms)
            motif_mask[res_idx, selected_atoms] = True
        motif_mask = motif_mask.bool()
        graph.motif_mask = motif_mask.bool()  # [n, 37]
        graph.x_motif = graph.coords_nm * graph.motif_mask[..., None]  # [n, 37, 3]
        graph.seq_motif_mask = motif_mask.sum(dim=-1).bool()  # [n]
        graph.seq_motif = graph.residue_type * graph.seq_motif_mask  # [n]
        return graph


class ExtractMotifCoordinatesTransform(T.BaseTransform):
    """
    Extracts motif coordinates and sequence information from a graph using the motif_mask.
    Adds x_motif, seq_motif_mask, and seq_motif attributes.
    """

    def __call__(self, graph: Data) -> Data:
        if not hasattr(graph, "motif_mask") or graph.motif_mask is None:
            raise ValueError(
                "motif_mask not found in graph. Apply MotifMaskTransform first."
            )
        graph.x_motif = graph.coords_nm * graph.motif_mask[..., None]  # [n, 37, 3]
        graph.seq_motif_mask = graph.motif_mask.sum(dim=-1).bool()  # [n]
        graph.seq_motif = graph.residue_type * graph.seq_motif_mask  # [n]
        return graph
