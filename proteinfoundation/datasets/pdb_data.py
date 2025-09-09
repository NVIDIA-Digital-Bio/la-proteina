import pathlib
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.utils.cluster_utils import (
    cluster_sequences,
    df_to_fasta,
    expand_cluster_splits,
    fasta_to_df,
    read_cluster_tsv,
    setup_clustering_file_paths,
    split_dataframe,
)
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

from graphein.ml.datasets import PDBManager
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.utils import download_pdb_multiprocessing


class PDBDataSelector:
    def __init__(
        self,
        data_dir: str,
        fraction: float = 1.0,
        min_length: int = None,
        max_length: int = None,
        molecule_type: str = None,
        experiment_types: List[str] = None,
        oligomeric_min: int = None,
        oligomeric_max: int = None,
        best_resolution: float = None,
        worst_resolution: float = None,
        has_ligands: List[str] = None,
        remove_ligands: List[str] = None,
        remove_non_standard_residues: bool = True,
        remove_pdb_unavailable: bool = True,
        labels: Optional[List[Literal["uniprot_id", "cath_code", "ec_number"]]] = None,
        remove_cath_unavailable: bool = False,
        exclude_ids: List[str] = None,
        exclude_ids_from_file: str = None,
        num_workers: int = 32,
    ):
        """
        Initialize the PDBDataSelector with the specified parameters.

        Args:
            data_dir (str): Directory path where the data is stored.
            fraction (float): Fraction of the data to be selected.
            min_length (int): Minimum length of the sequences to be included.
            max_length (int): Maximum length of the sequences to be included.
            molecule_type (str): Type of the molecule (e.g., "protein", "DNA", "RNA").
            experiment_types (List[str]): List of experiment types to be included.
            oligomeric_min (int): Minimum oligomeric state of the structures to be included.
            oligomeric_max (int): Maximum oligomeric state of the structures to be included.
            best_resolution (float): Best resolution threshold for the structures to be included.
            worst_resolution (float): Worst resolution threshold for the structures to be included.
            has_ligands (List[str]): List of ligands that must be present in the structures.
            remove_ligands (List[str]): List of ligands to be removed from the structures.
            remove_non_standard_residues (bool): Whether to remove non-standard residues from the structures.
            remove_pdb_unavailable (bool): Whether to remove structures that are not available in the PDB.
            labels (Optional[List[Literal["uniprot_id", "cath_code", "ec_number"]]], optional): A list of names corresponding to metadata labels that should be included in PDB manager dataframe.
                Defaults to ``None``.
            remove_cath_unavailable (bool): Whether to remove structures that don't have CATH labels.
            exclude_ids (List[str]): List of PDB IDs to be excluded from the selection.
            exclude_ids_from_file (str, optional): Path to a txt file containing IDs to be excluded from the selection.
            num_workers (int): Number of workers for parallel processing. Defaults to 32.

        Raises:
            AssertionError: If the sum of train_val_test fractions is not equal to 1.
            TypeError: If split_time_frames does not contain valid dates for np.datetime64 format.
        """
        self.database = "pdb"
        self.data_dir = pathlib.Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.fraction = fraction
        self.molecule_type = molecule_type
        self.experiment_types = experiment_types
        self.oligomeric_min = oligomeric_min
        self.oligomeric_max = oligomeric_max
        self.best_resolution = best_resolution
        self.worst_resolution = worst_resolution
        self.has_ligands = has_ligands
        self.remove_ligands = remove_ligands
        self.remove_non_standard_residues = remove_non_standard_residues
        self.remove_pdb_unavailable = remove_pdb_unavailable
        self.min_length = min_length
        self.max_length = max_length
        self.exclude_ids = exclude_ids
        self.exclude_ids_from_file = exclude_ids_from_file
        self.labels = labels
        self.remove_cath_unavailable = remove_cath_unavailable
        self.num_workers = num_workers
        self.df_data = None

    def create_dataset(self) -> pd.DataFrame:
        """Filter PDB data based on metadata and constraints and return a dataframe with the selected datapoints.

        Returns:
            pd.DataFrame: dataframe containing all datapoints that satisfy the criteria.
        """
        # lazy init
        if self.df_data:
            return self.df_data

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing PDBManager in {self.data_dir}...")
        pdb_manager = PDBManager(root_dir=self.data_dir, labels=self.labels)

        num_chains = len(pdb_manager.df)
        logger.info(f"Starting with: {num_chains} chains")

        # subsample dataframe based on provided fraction
        if self.fraction != 1.0:
            logger.info(f"Subsampling data to {self.fraction} fraction")
            pdb_manager.df = pdb_manager.df.sample(frac=self.fraction)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.experiment_types:
            logger.info(
                f"Removing chains that are not in one of the following experiment types: {self.experiment_types}"
            )
            pdb_manager.experiment_types(self.experiment_types, update=True)

        if self.max_length:
            logger.info(f"Removing chains longer than {self.max_length}...")
            pdb_manager.length_shorter_than(self.max_length, update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.min_length:
            logger.info(f"Removing chains shorter than {self.min_length}...")
            pdb_manager.length_longer_than(self.min_length, update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.molecule_type:
            logger.info(
                f"Removing chains molecule types not in selection: {self.molecule_type}..."
            )
            pdb_manager.molecule_type(self.molecule_type, update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        logger.info(
            f"Removing chains oligomeric state not in selection: {self.oligomeric_min} - {self.oligomeric_max}..."
        )
        if self.oligomeric_min:
            pdb_manager.oligomeric(self.oligomeric_min, "greater", update=True)
        if self.oligomeric_max:
            pdb_manager.oligomeric(self.oligomeric_max, "less", update=True)
        logger.info(f"{len(pdb_manager.df)} chains remaining")

        logger.info(
            f"Removing chains with resolution not in selection: {self.best_resolution} - {self.worst_resolution}..."
        )
        if self.worst_resolution:
            pdb_manager.resolution_better_than_or_equal_to(
                self.worst_resolution, update=True
            )
        if self.best_resolution:
            pdb_manager.resolution_worse_than_or_equal_to(
                self.best_resolution, update=True
            )
        logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.remove_ligands:
            logger.info(
                f"Removing chains with ligands in selection: {self.remove_ligands}..."
            )
            pdb_manager.has_ligands(self.remove_ligands, inverse=True, update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.has_ligands:
            logger.info(
                f"Removing chains without ligands in selection: {self.has_ligands}..."
            )
            pdb_manager.has_ligands(self.has_ligands, update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        if self.remove_non_standard_residues:
            logger.info("Removing chains with non-standard residues...")
            pdb_manager.remove_non_standard_alphabet_sequences(update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")
        if self.remove_pdb_unavailable:
            logger.info("Removing chains with PDB unavailable...")
            pdb_manager.remove_unavailable_pdbs(update=True)
            logger.info(f"{len(pdb_manager.df)} chains remaining")
        if self.remove_cath_unavailable:
            logger.info("Removing chains with cath code unavailable...")
            mask = ~pdb_manager.df["cath_code"].isna()
            pdb_manager.df = pdb_manager.df[mask]
            logger.info(f"{len(pdb_manager.df)} chains remaining")

        all_exclude_ids = set()
        # Add IDs from direct list if present
        if self.exclude_ids:
            all_exclude_ids.update(self.exclude_ids)
            # Add IDs from file if present
        if self.exclude_ids_from_file:
            with open(self.exclude_ids_from_file, "r") as f:
                file_ids = {line.strip() for line in f if line.strip()}
            all_exclude_ids.update(file_ids)

        logger.info(f"Removing excluded chains ({len(all_exclude_ids)} gathered)")

        mask = ~pdb_manager.df["id"].isin(all_exclude_ids)
        pdb_manager.df = pdb_manager.df[mask]
        logger.info(f"{len(pdb_manager.df)} chains remaining")
        self.df_data = pdb_manager.df
        return self.df_data


class PDBDataSplitter:
    def __init__(
        self,
        df_data: pd.DataFrame = None,
        data_dir: str = None,
        train_val_test: List[float] = [0.8, 0.15, 0.05],
        split_type: Literal["random", "sequence_similarity"] = "random",
        split_sequence_similarity: Optional[int] = None,
        overwrite_sequence_clusters: Optional[bool] = False,
    ) -> None:
        """Initialise DataSplitter object for splitting data based on arguments into train, val and test set.

        Args:
            df_data (pd.DataFrame, optional): DataFrame containing the sample IDs and metadata. Defaults to None.
            data_dir (str, optional): directory contain the sample files. Defaults to None.
            train_val_test (List[float], optional): proportion of train, validation and test set. Defaults to [0.8, 0.15, 0.05].
            split_type (Literal["random", "sequence_similarity"], optional): If the dataset should be
                split randomly into train, val and test or via sequence similarity clustering.
                Defaults to "random".
            split_sequence_similarity (Optional[float], optional): if split_type == "sequence_similarity",
                which sequence similarity threshold should be chosen (0.3 means 30% sequence similarity
                clustering). Defaults to None.
            overwrite_sequence_clusters (Optional[bool], optional): if split_type == "sequence_similarity", if previously
                generated clusters (if present with the same sequence similarity threshold) should be overwritten
                or reused. Defaults to False (reuse).
        """

        self.df_data = df_data
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.split_type = split_type
        self.split_sequence_similarity = split_sequence_similarity
        self.overwrite_sequence_clusters = overwrite_sequence_clusters
        self.splits = ["train", "val", "test"]
        self.dfs_splits = None
        self.clusterid_to_seqid_mappings = None

    def split_data(self, df_data: pd.DataFrame, file_identifier: str) -> Dict:
        """
        Splits the dataframe into train, val and test splits based on the split type and sampling mode.

        Args:
            df_data (pd.DataFrame): dataframe containing the data to be split

        Returns:
            dfs_splits (Dict): dictionary containing the train/val/test splits of the dataframe.
        """
        if self.split_type == "random":
            logger.info(
                f"Splitting dataset via random split into {self.train_val_test}..."
            )
            self.dfs_splits = split_dataframe(df_data, self.splits, self.train_val_test)
            self.clusterid_to_seqid_mappings = None

        elif self.split_type == "sequence_similarity":
            logger.info(
                f"Splitting dataset via sequence-similarity split into {self.train_val_test}..."
            )
            logger.info(
                f"Using {self.split_sequence_similarity} sequence similarity for split"
            )
            input_fasta_filepath, cluster_fasta_filepath, cluster_tsv_filepath = (
                setup_clustering_file_paths(
                    self.data_dir,
                    file_identifier,
                    self.split_sequence_similarity,
                )
            )

            if not input_fasta_filepath.exists() or self.overwrite_sequence_clusters:
                logger.info("Retrieving sequences and writing them to fasta file...")
                df_to_fasta(df=df_data, output_file=input_fasta_filepath)

            if not cluster_fasta_filepath.exists() or self.overwrite_sequence_clusters:
                logger.info("Clustering sequences via mmseqs2...")
                cluster_sequences(
                    fasta_input_filepath=input_fasta_filepath,
                    cluster_output_filepath=cluster_fasta_filepath,
                    min_seq_id=self.split_sequence_similarity,
                    overwrite=self.overwrite_sequence_clusters,
                )
            # read representative sequences in
            df_cluster_reps = fasta_to_df(cluster_fasta_filepath)
            seq_ids = df_cluster_reps["id"].to_numpy().tolist()
            # only select sequence representatives from original df to generate random splits of clusters
            df_sequences_reps = df_data.loc[df_data.id.isin(seq_ids)]
            splits = split_dataframe(
                df_sequences_reps, self.splits, self.train_val_test
            )
            # construct cluster_dict to map from cluster representative to all sequence ids in cluster
            clusterid_to_seqid_mapping = read_cluster_tsv(cluster_tsv_filepath)
            # use cluster dict to extend splits from cluster representatives to all sequence ids included in these clusters
            self.dfs_splits, self.clusterid_to_seqid_mappings = expand_cluster_splits(
                cluster_rep_splits=splits,
                clusterid_to_seqid_mapping=clusterid_to_seqid_mapping,
            )
        return (
            self.dfs_splits,
            self.clusterid_to_seqid_mappings,
        )


class PDBDataset(Dataset):
    def __init__(
        self,
        pdb_codes: List[str],
        chains: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        format: Literal["mmtf", "pdb", "cif", "ent"] = "cif",
        in_memory: bool = False,
        file_names: Optional[List[str]] = None,
        num_workers: int = 64,
    ):
        """
        Args:
            pdb_codes (List[str]): List of PDB codes or identifiers specific to your
                filenames for the structures to load.
            chains (List[str], optional): List of chains to load for each PDB code.
                Defaults to None.
            data_dir (str, optional): Path to the data directory. Defaults to None.
            transform (Callable, optional): Transform to apply to each
                example. Defaults to None.
            format (str, optional): Format to save structures in. Can be one of
                "mmtf", "pdb", "cif" or "ent". Defaults to "cif".
            in_memory (bool, optional): Whether to load data into memory.
                Defaults to False.
            file_names (List[str], optional): How to name the processed data files. By default '{pdb_code}.pt'.
            num_workers (int, optional): How many workers to use for pdb data downloads.
                Defaults to 8.
        """
        self.database = "pdb"
        self.pdb_codes = [pdb.lower() for pdb in pdb_codes]
        self.chains = chains
        self.format = format
        self.data_dir = pathlib.Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.in_memory = in_memory
        self.file_names = file_names
        self.num_workers = num_workers
        self.transform = transform
        self.sequence_id_to_idx = None

        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [torch.load(self.processed_dir / f) for f in tqdm(file_names)]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Data:
        """Return PyTorch Geometric Data object for a given index.

        Args:
            idx (int): Index to retrieve.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        if self.in_memory:
            graph = self.data[idx]
        else:
            if self.file_names is not None:
                fname = f"{self.file_names[idx]}.pt"
            elif self.chains is not None:
                fname = f"{self.pdb_codes[idx]}_{self.chains[idx]}.pt"
            else:
                fname = f"{self.pdb_codes[idx]}.pt"

            graph = torch.load(self.data_dir / "processed" / fname, weights_only=False)

        # reorder coords to be in OpenFold and not PDB convention
        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

        if self.transform:
            graph = self.transform(graph)

        return graph


class PDBLightningDataModule(BaseLightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        dataselector: Optional[PDBDataSelector] = None,
        datasplitter: Optional[PDBDataSplitter] = None,
        in_memory: bool = False,
        format: Literal["mmtf", "pdb", "cif", "ent"] = "cif",
        overwrite: bool = False,
        store_het: bool = False,
        store_bfactor: bool = True,
        # arguments for BaseLightningDataModule
        batch_padding: bool = True,
        sampling_mode: Literal["random", "cluster-random", "cluster-reps"] = "random",
        transforms: Optional[List[Callable]] = None,
        pre_transforms: Optional[List[Callable]] = None,
        pre_filters: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
        **kwargs,
    ):
        """Initializes the PDBLightningDataModule.

        Args:
            data_dir (str, optional): directory where PDB data should be stored.
                Default to None.
            dataselector (PDBDataSelector, optional): Selector for PDB data.
                Defaults to None.
            datasplitter (PDBDataSplitter, optional): Splitter for PDB data
                to create train/val/test splits. Defaults to None.
            in_memory (bool, optional): Whether to load the entire dataset into
                memory. Defaults to False.
            format (str, optional): Format to save structures in. Can be one of
                "mmtf", "pdb", "cif" or "ent". Defaults to "cif".
            overwrite (bool, optional): Whether to overwrite existing processed
                data. Defaults to False.
            store_het (bool, optional): Whether to store heteroatoms in the processed data.
                Defaults to False.
            store_bfactor (bool, optional): Whether to store B factors in the processed data.
                Defaults to True.
            batch_padding (bool, optional): Whether batches should be padded to a dense representation
                with the length being either a pre-specified max length or the maximum length of the
                sample in the batch (base PyTorch batch) or whether a sparse representation should be
                used (PyG batch). Defaults to True (base PyTorch batch).
            sampling_mode (Literal["random", "cluster-random", "cluster-reps"], optional): How the data should be
                sampled from the dataset later on:
                - "random": Select a random sequence and ignore clusters.
                - "cluster-random": Select a random sequence from each cluster. Keep all samples for each cluster.
                - "cluster-reps": Select the cluster representative from each cluster. Only keep the representative for each cluster.
                  Defaults to "random".
            transforms (List[Callable]): List of transforms applied to each example.
            pre_transforms (Callable): Transform applied to each example before processing.
            pre_filters (Callable): Filter applied to each example before processing.
            batch_size (int, optional): Batch size used for dataloaders. Defaults to 32.
            num_workers (int, optional): Number of workers used for dataloading. Defaults to 32.
            pin_memory (bool, optional): Whether memory should be pinned. Defaults to False.
        """
        super().__init__(
            batch_padding=batch_padding,
            sampling_mode=sampling_mode,
            transforms=transforms,
            pre_transforms=pre_transforms,
            pre_filters=pre_filters,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.data_dir = pathlib.Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.dataselector = dataselector
        self.datasplitter = datasplitter
        self.sampling_mode = sampling_mode
        self.format = format
        self.overwrite = overwrite
        self.in_memory = in_memory
        self.store_het = store_het
        self.store_bfactor = store_bfactor
        self.df_data = None
        self.dfs_splits = None
        self.clusterid_to_seqid_mappings = None
        self.file_names = None

    def prepare_data(self):
        if self.dataselector:
            file_identifier = self._get_file_identifier(self.dataselector)
            df_data_name = f"{file_identifier}.csv"
            if not self.overwrite and (self.data_dir / df_data_name).exists():
                logger.info(
                    f"{df_data_name} already exists, skipping data selection and processing stage."
                )
            else:
                logger.info(f"{df_data_name} does not exist yet, creating dataset now.")
                df_data = self.dataselector.create_dataset()
                logger.info(
                    f"Dataset created with {len(df_data)} entries. Now downloading structure data..."
                )
                self._download_structure_data(df_data["pdb"].tolist())
                # process pdb files into seperate chains and save processed objects as .pt files
                self._process_structure_data(
                    df_data["pdb"].tolist(), df_data["chain"].tolist()
                )

                # save df_data to disk for later use (in splitting, dataloading etc)
                logger.info(f"Saving dataset csv to {df_data_name}")
                df_data.to_csv(self.data_dir / df_data_name, index=False)

        else:  # user-provided dataset
            df_data_name = f"{self.data_dir.name}.csv"
            if not self.overwrite and (self.data_dir / df_data_name).exists():
                logger.info(
                    f"{df_data_name} already exists, skipping data selection and processing stage."
                )
            else:
                logger.info(f"{df_data_name} does not exist yet, creating dataset now.")
                df_data = self._load_pdb_folder_data(self.raw_dir)
                # process pdb files into seperate chains and save processed objects as .pt files
                self._process_structure_data(
                    pdb_codes=df_data["pdb"].tolist(),
                    chains=None,
                )
                # save df_data to disk for later use (in splitting, dataloading etc)
                logger.info(f"Saving dataset csv to {df_data_name}")
                df_data.to_csv(self.data_dir / df_data_name, index=False)
            
    def _load_pdb_folder_data(self, data_dir: pathlib.Path) -> pd.DataFrame:
        """
        Load PDB files from a folder and create a DataFrame with filenames.
        
        Args:
            data_dir (pathlib.Path): Path to the directory containing PDB files
            
        Returns:
            pd.DataFrame: DataFrame with 'pdb' column containing filenames
        """
        # Get all files with the specified format extension
        pdb_files = list(data_dir.glob(f"*.{self.format}"))
        
        # Create DataFrame with filenames
        df_data = pd.DataFrame({
            'pdb': [pdb_file.stem for pdb_file in pdb_files],
            'id': [pdb_file.stem for pdb_file in pdb_files],
        })
        
        if len(df_data) == 0:
            raise ValueError(f"No files with extension .{self.format} found in {data_dir}")
            
        logger.info(f"Found {len(df_data)} {self.format} files in {data_dir}")
        
        return df_data

    def _get_file_identifier(self, ds):
        file_identifier = (
            f"df_pdb_f{ds.fraction}_minl{ds.min_length}_maxl{ds.max_length}_mt{ds.molecule_type}"
            f"_et{''.join(ds.experiment_types) if ds.experiment_types else ''}"
            f"_mino{ds.oligomeric_min}_maxo{ds.oligomeric_max}"
            f"_minr{ds.best_resolution}_maxr{ds.worst_resolution}"
            f"_hl{''.join(ds.has_ligands) if ds.has_ligands else ''}"
            f"_rl{''.join(ds.remove_ligands) if ds.remove_ligands else ''}"
            f"_rnsr{ds.remove_non_standard_residues}_rpu{ds.remove_pdb_unavailable}"
            f"_l{''.join(ds.labels) if ds.labels else ''}"
            f"_rcu{ds.remove_cath_unavailable}"
        )
        return file_identifier

    def setup(self, stage: Optional[str] = None):
        """Split data into train, val and test sets and create dataset objects.

        Args:
            stage (Optional[str], optional): Which dataset should be created (train, val or test). Defaults to None.
        """
        # load dataframe with metadata from disk
        if not self.df_data:
            if self.dataselector:
                file_identifier = self._get_file_identifier(self.dataselector)
            else:
                file_identifier = self.data_dir.name

            df_data_name = f"{file_identifier}.csv"
            logger.info(f"Loading dataset csv from {df_data_name}")
            self.df_data = pd.read_csv(self.data_dir / df_data_name)

        # split the dataset into train, val and test and set attributes that are used for dataset creation
        (self.dfs_splits, self.clusterid_to_seqid_mappings) = (
            self.datasplitter.split_data(self.df_data, file_identifier)
        )

        # create appropriate datasets based on the selected stage
        if stage == "fit" or stage is None:
            self.train_ds = self.train_dataset()
            self.val_ds = self.val_dataset()
        elif stage == "test":
            self.test_ds = self.test_dataset()

    def _process_structure_data(self, pdb_codes, chains):
        """Process raw data sequentially instead of using multiprocessing."""
        if chains is not None:
            index_pdb_tuples = [
                (i, pdb, chains[i])
                for i, pdb in enumerate(pdb_codes)
                if not (self.processed_dir / f"{pdb}_{chains[i]}.pt").exists()
            ]
        else:
            index_pdb_tuples = [
                (i, pdb)
                for i, pdb in enumerate(pdb_codes)
                if not (self.processed_dir / f"{pdb}.pt").exists()
            ]

        file_names = []
        for tuple_ in tqdm(index_pdb_tuples, desc="Processing structures", unit="file"):
            result = self._load_and_process_pdb(tuple_)
            if result is not None:
                file_names.append(result)
        
        logger.info("Completed processing.")
        return file_names

    def _load_and_process_pdb(
        self, index_pdb_tuple: Union[Tuple[int, str], Tuple[int, str, str]]
    ) -> Optional[str]:
        """
        Load and process a PDB file, converting it to a PyTorch Geometric graph.

        This function takes a tuple containing an index and a PDB code (and optionally a chain),
        loads the corresponding PDB file, processes it into a graph, and saves the result.

        Args:
            index_pdb_tuple (Union[Tuple[int, str], Tuple[int, str, str]]): A tuple containing:
                - index (int): The index of the PDB file in the list.
                - pdb (str): The PDB code.
                - chains (str, optional): The chains to process. If not provided, all chains are processed.

        Returns:
            Optional[str]: The filename of the saved processed graph, or None if processing failed.

        Raises:
            FileNotFoundError: If the PDB file is not found in the raw directory.
        """
        try:
            if len(index_pdb_tuple) == 3:
                i, pdb, chains = index_pdb_tuple
            elif len(index_pdb_tuple) == 2:
                i, pdb = index_pdb_tuple
                chains = "all"
            else:
                raise ValueError("index_pdb_tuple must have 2 or 3 elements")

            path = self.raw_dir / f"{pdb}.{self.format}"
            if path.exists():
                path = str(path)
            elif path.with_suffix("." + self.format + ".gz").exists():
                path = str(path.with_suffix("." + self.format + ".gz"))
            else:
                raise FileNotFoundError(
                    f"{pdb} not found in raw directory. Are you sure it's downloaded and has the format {self.format}?"
                )

            fill_value_coords = 1e-5
            graph = protein_to_pyg(
                path=path,
                chain_selection=chains,
                keep_insertions=True,
                store_het=self.store_het,
                store_bfactor=self.store_bfactor,
                fill_value_coords=fill_value_coords,
            )

        except Exception as e:
            logger.warning(f"Error processing {pdb} {chains}: {e}")
            return None
        fname = f"{pdb}.pt" if chains == "all" else f"{pdb}_{chains}.pt"

        graph.id = fname.split(".")[0]
        coord_mask = graph.coords != fill_value_coords
        graph.coord_mask = coord_mask[..., 0]
        graph.residue_type = torch.tensor(
            [resname_to_idx[residue] for residue in graph.residues]
        ).long()
        graph.database = "pdb"
        graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
        )
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

        if self.pre_transform:
            graph = self.pre_transform(graph)

        if self.pre_filter:
            if self.pre_filter(graph) is not True:
                return None

        torch.save(graph, self.processed_dir / fname)
        return fname

    def _download_structure_data(self, pdb_codes) -> None:
        if pdb_codes is not None:
            to_download = (
                pdb_codes
                if self.overwrite
                else [
                    pdb
                    for pdb in pdb_codes
                    if not (
                        (self.raw_dir / f"{pdb}.{self.format}").exists()
                        or (self.raw_dir / f"{pdb}.{self.format}.gz").exists()
                    )
                ]
            )
            to_download = list(set(to_download))
            # Determine whether to download raw structures
            if to_download:
                logger.info(
                    f"Downloading {len(to_download)} structures to {self.processed_dir}"
                )
                file_format = (
                    self.format[:-3] if self.format.endswith(".gz") else self.format
                )
                # calculate number of downloads per worker
                chunksize = (
                    len(to_download) // self.num_workers + 1
                )  # +1 handles edge case where num_workers > len(to_download)
                download_pdb_multiprocessing(
                    to_download,
                    self.raw_dir,
                    format=file_format,
                    max_workers=self.num_workers,
                    chunksize=chunksize,
                )
            else:
                logger.info(
                    f"No structures to download, all {len(pdb_codes)} structure files already present"
                )

    def _get_dataset(self, split: Literal["train", "val", "test"]) -> PDBDataset:
        """Initialises a dataset for a given split.

        Args:
            split Literal["train", "val", "test"]: Split to initialise.

        Returns:
            PDBCompDataset: initialised dataset for one split
        """
        df_split = self.dfs_splits[split]
        self.clusterid_to_seqid_mappings = self.clusterid_to_seqid_mappings
        pdb_codes = df_split["pdb"].tolist()
        # Check if 'chain' column exists in the DataFrame
        if 'chain' in df_split.columns:
            chains = df_split["chain"].tolist()
            file_names = [f"{pdb}_{chain}" for pdb, chain in zip(pdb_codes, chains)]
        else:
            chains = None
            file_names = [f"{pdb}" for pdb in pdb_codes]

        return PDBDataset(
            pdb_codes=pdb_codes,
            chains=chains,
            data_dir=self.data_dir,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            file_names=file_names,
            num_workers=self.num_workers,
        )