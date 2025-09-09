# inspired from https://github.com/a-r-j/graphein/blob/master/graphein/ml/datasets/pdb_data.py

import math
import pathlib
import random
import shutil
import subprocess
from typing import Dict, List, Literal, Tuple

import pandas as pd
import torch
import torch_geometric
from graphein.utils.dependencies import is_tool
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from torch.utils.data import Sampler


@rank_zero_only
def log_info(msg):
    logger.info(msg)


class ClusterSampler(Sampler):
    def __init__(
        self,
        dataset: torch_geometric.data.Dataset,
        clusterid_to_seqid_mapping: Dict[str, List[str]],
        sampling_mode: Literal["cluster-random", "cluster-reps"],
        shuffle: bool = True,
        drop_last: bool = False,
        dimer_mode: bool = False,
    ):
        """
        Initializes the ClusterSampler for selecting sequences during training.

        Args:
            dataset (torch_geometric.data.Dataset): The dataset object.
            clusterid_to_seqid_mapping (Dict[str, List[str]]): Dictionary holding cluster names and corresponding sequence IDs.
            sampling_mode (Literal["cluster-random", "cluster-reps"]): The sampling mode to use.
                - "cluster-random": Select a random sequence from each cluster.
                - "cluster-reps": Select the representative sequence from each cluster.
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the indices.
            drop_last (bool, optional): If ``True``, then the sampler will drop the tail of the data to make it
                evenly divisible across the number of replicas. If ``False``, the sampler will add extra indices to
                make the data evenly divisible across the replicas. Default: ``False``.
            dimer_mode (bool, optional): If ``True``, the sampler will have dimerspecific IDS, but for passing
                to the dataset, it will remove the dimer ID to have a normal AFDB ID. Default: ``False``.
        """
        self.dataset = dataset
        self.clusterid_to_seqid_mapping = clusterid_to_seqid_mapping
        self.cluster_names = list(clusterid_to_seqid_mapping.keys())
        self.sampling_mode = sampling_mode
        if dataset.database == "pdb" or dataset.database == "scop":  # PDBDataset
            self.sequence_id_to_idx = {
                fname.split(".")[0]: i for i, fname in enumerate(dataset.file_names)
            }
        elif dataset.database == "pinder":
            self.sequence_id_to_idx = dataset.pinder_id_to_idx
        else:  # FoldCompDataset
            self.sequence_id_to_idx = dataset.protein_to_idx
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.log_clusters = True
        self.num_replicas = None
        self.dimer_mode = False

    def __iter__(self):
        """Iterate over clusters in dataset and yield samples depending on sampling_mode."""
        # set logging to true so that first sample in epoche gets logged
        self.log_clusters = True
        # setup distributed/non-distributed backend
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = None
            self.rank = 0
            logger.info(
                f"Distributed sampler is not initialized, assuming single-device setup."
            )

        if self.num_replicas is not None:
            self.num_samples = math.ceil(
                len(self.cluster_names) * 1.0 / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
            # Distributed mode, deterministically shuffle
            indices = torch.randperm(len(self.cluster_names)).tolist()

            # drop samples to make it evenly divisible
            if self.drop_last:
                indices_to_keep = self.total_size - self.num_replicas
                indices = indices[:indices_to_keep]
            # add extra samples to make it evenly divisible
            else:
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]

            # subsample
            indices = indices[self.rank : self.total_size : self.num_replicas]
            if self.sampling_mode == "cluster-reps":
                # Assumes that cluster_names are the IDs of the representative (longest) sequences (true for mmseqs2 clusters)
                for cluster_name_idx in indices:
                    cluster_name = self.cluster_names[cluster_name_idx]
                    if self.dimer_mode:
                        # remove everything before the first underscore, keep the rest
                        cluster_name = cluster_name.split("_", 1)[1]
                    yield self.sequence_id_to_idx[cluster_name]
            elif self.sampling_mode == "cluster-random":
                for cluster_name_idx in indices:
                    cluster_name = self.cluster_names[cluster_name_idx]
                    sequences = self.clusterid_to_seqid_mapping[cluster_name]
                    sequence_id = random.choice(sequences)
                    if self.log_clusters:
                        # log first sampling
                        logger.info(
                            f"First cluster sampling: sampling {sequence_id} from cluster {cluster_name}, rank {self.rank}"
                        )
                        self.log_clusters = False
                    if self.dimer_mode:
                        # remove everything before the first underscore, keep the rest
                        sequence_id = sequence_id.split("_", 1)[1]
                    yield self.sequence_id_to_idx[sequence_id]
            else:
                raise ValueError(
                    f"Unknown cluster sampling mode {self.sampling_mode} for ClusterSampler, only 'cluster-random' and 'cluster-reps' supported"
                )
        else:
            # Non-distributed mode
            if self.shuffle:
                random.shuffle(self.cluster_names)
            if self.sampling_mode == "cluster-reps":
                # Assumes that cluster_names are the IDs of the representative (longest) sequences (true for mmseqs2 clusters)
                for cluster_name in self.cluster_names:
                    if self.dimer_mode:
                        # remove everything before the first underscore, keep the rest
                        cluster_name = cluster_name.split("_", 1)[1]
                    yield self.sequence_id_to_idx[cluster_name]
            elif self.sampling_mode == "cluster-random":
                for cluster_name in self.cluster_names:
                    sequences = self.clusterid_to_seqid_mapping[cluster_name]
                    sequence_id = random.choice(sequences)
                    if self.log_clusters:
                        # log first sampling
                        logger.info(
                            f"First cluster sampling: sampling {sequence_id} from cluster {cluster_name}"
                        )
                        self.log_clusters = False
                    if self.dimer_mode:
                        # remove everything before the first underscore, keep the rest
                        sequence_id = sequence_id.split("_", 1)[1]
                    yield self.sequence_id_to_idx[sequence_id]
            else:
                raise ValueError(
                    f"Unknown cluster sampling mode {self.sampling_mode} for ClusterSampler, only 'cluster-random' and 'cluster-reps' supported"
                )

    def __len__(self):
        if self.num_replicas is not None:
            return self.num_samples
        else:
            return len(self.cluster_names)


def split_dataframe(
    df: pd.DataFrame,
    splits: List[str],
    ratios: List[float],
    leftover_split: int = 0,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split a DataFrame into multiple parts based on specified split ratios.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        splits (List[str]): Names of the resulting splits.
        ratios (List[float]): Ratios to split df into. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover rows to.
            Defaults to 0.
        seed (int): Random seed for shuffling. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to
            DataFrame splits.

    Raises:
        AssertionError: If len(splits) != len(ratios) or sum(ratios) != 1.
    """
    assert len(splits) == len(ratios), "Number of splits must equal number of ratios"
    assert sum(ratios) == 1, "Split ratios must sum to 1"

    # Calculate size of each split
    split_sizes = [int(len(df) * ratio) for ratio in ratios]

    # Assign leftover rows to specified split
    split_sizes[leftover_split] += len(df) - sum(split_sizes)

    # Shuffle DataFrame rows
    df = df.sample(frac=1, random_state=seed)

    # Split DataFrame into parts
    split_dfs = {}
    start = 0
    for split, size in zip(splits, split_sizes):
        split_dfs[split] = df.iloc[start : start + size]
        start += size

    return split_dfs


def merge_dataframe_splits(
    df1: pd.DataFrame, df2: pd.DataFrame, list_columns: List[str]
) -> pd.DataFrame:
    """
    Merge two DataFrame splits on all columns except 'split'.

    Args:
        df1 (pd.DataFrame): First DataFrame split to merge.
        df2 (pd.DataFrame): Second DataFrame split to merge.
        list_columns (List[str]): Columns containing lists to convert to tuples.

    Returns:
        pd.DataFrame: Merged DataFrame containing rows in both splits.
    """
    # Convert list columns to tuples for merging
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(tuple)

    # Merge the two DataFrames
    merge_cols = [c for c in df1.columns if c != "split"]
    merged_df = pd.merge(df1, df2, on=merge_cols, how="inner")

    # Convert tuple columns back to lists
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(list)

    return merged_df


def cluster_sequences(
    fasta_input_filepath: str,
    cluster_output_filepath: str = None,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    overwrite: bool = False,
    silence_mmseqs_output: bool = True,
    efficient_linclust: bool = False,
    mmseqs_exec: str = None,
) -> None:
    """
    Cluster protein sequences in a DataFrame using MMseqs2.

    Args:
        fasta_input_file (str): Fasta File path containing protein sequences.
        cluster_output_filepath (str): Path to write clustering results. If None, defaults to
            "cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta".
        min_seq_id (float): Minimum sequence identity for clustering. Defaults to 0.3.
        coverage (float): Minimum coverage for clustering. Defaults to 0.8.
        overwrite (bool): Whether to overwrite existing cluster file. Defaults to False.
        silence_mmseqs_output (bool): Whether to silence MMseqs2 output. Defaults to True.
        efficient_linclust (bool): Whether to use efficient linclust for clustering for large datasets. Defaults to False.
        mmseqs_exec (str): Path to the mmseqs2 executable. Defaults to None. If not provided, the function will check if mmseqs2 is installed.
    """
    if cluster_output_filepath is None:
        cluster_output_filepath = f"cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta"

    cluster_fasta_path = pathlib.Path(cluster_output_filepath)
    cluster_tsv_path = cluster_fasta_path.with_suffix(".tsv")

    if not cluster_fasta_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_fasta_path.exists() and overwrite:
            cluster_fasta_path.unlink()

    if not cluster_tsv_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_tsv_path.exists() and overwrite:
            cluster_tsv_path.unlink()

        # Run MMseqs2 clustering
        if mmseqs_exec is None and not is_tool("mmseqs"):
            logger.error(
                "MMseqs2 not found. Please install it: conda install -c conda-forge -c bioconda mmseqs2"
            )

        mmseqs_exec = "mmseqs" if mmseqs_exec is None else mmseqs_exec
        if (
            efficient_linclust
        ):  # use efficient linclust algorithm that cales linearly with input size
            cmd = f"{mmseqs_exec} easy-linclust {fasta_input_filepath} pdb_cluster tmp --min-seq-id {min_seq_id} -c {coverage} --cov-mode 1"
        else:  # use standard cascaded clustering algorithm
            cmd = f"{mmseqs_exec} easy-cluster {fasta_input_filepath} pdb_cluster tmp --min-seq-id {min_seq_id} -c {coverage} --cov-mode 1"
        if silence_mmseqs_output:
            subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)
        else:
            subprocess.run(cmd.split())
        # Rename output file
        shutil.move("pdb_cluster_rep_seq.fasta", cluster_fasta_path)
        shutil.move("pdb_cluster_cluster.tsv", cluster_tsv_path)


def split_sequence_clusters(
    df, splits, ratios, leftover_split=0, seed=42
) -> Dict[str, pd.DataFrame]:
    """
    Split clustered sequences into train/val/test sets.

    Args:
        df (pd.DataFrame): DataFrame with clustered sequences.
        splits (List[str]): Names of splits, e.g. ["train", "val", "test"].
        ratios (List[float]): Ratios for each split. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover sequences.
            Defaults to 0.
        seed (int): Random seed. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to DataFrames that contain randomly-split representative sequences.
    """
    # Split clusters into subsets
    cluster_splits = split_dataframe(df, splits, ratios, leftover_split, seed)
    # Get representative sequences for each split
    split_dfs = {}
    for split, cluster_df in cluster_splits.items():
        rep_seqs = cluster_df.representative_sequences()
        split_dfs[split] = rep_seqs

    return split_dfs


def expand_cluster_splits(
    cluster_rep_splits: Dict[str, pd.DataFrame],
    clusterid_to_seqid_mapping: Dict[str, List[str]],
    use_modin: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Expand the cluster representative splits to full cluster splits based on the provided cluster dictionary.

    Args:
        cluster_rep_splits: A dictionary containing DataFrames for each split (e.g., 'train', 'val', 'test').
            Each DataFrame should have an 'id' column representing the cluster representative IDs.
        clusterid_to_seqid_mapping: A dictionary mapping cluster representative IDs to their corresponding cluster member IDs.
        use_modin (bool): Whether to use Modin for dataframe operations, useful for big datasets. Defaults to False.

    Returns:
        A new dictionary of DataFrames with expanded 'id' columns based on the cluster dictionary.
        The 'id' column in the original DataFrames is replaced with the corresponding cluster member IDs.
        If df_sequences is provided, the additional columns from df_sequences are added to the resulting DataFrames.

    """
    full_cluster_splits = {}
    split_clusterid_to_seqid_mapping = {}

    for split_name, split_df in cluster_rep_splits.items():
        # Create a dictionary to store the cluster members for the current split
        split_cluster_members = {}

        for rep_id in split_df["id"]:
            if rep_id in clusterid_to_seqid_mapping:
                split_cluster_members[rep_id] = clusterid_to_seqid_mapping[rep_id]
            else:
                logger.warning(
                    f"ID {rep_id} is a representative in the splits, but not in the cluster_dicts"
                )

        # Create a DataFrame with the cluster representative IDs and their corresponding cluster member IDs for the current split
        split_cluster_members_df = pd.DataFrame(
            [
                (rep_id, member_id)
                for rep_id, member_ids in split_cluster_members.items()
                for member_id in member_ids
            ],
            columns=["cluster_id", "id"],
        )
        # Split the 'id' column into 'pdb' and 'chain' columns
        if len(split_cluster_members_df) > 0:
            split_cluster_members_df[["pdb", "chain"]] = split_cluster_members_df[
                "id"
            ].str.split("_", n=1, expand=True)
        # Add the expanded DataFrame to the dictionary
        full_cluster_splits[split_name] = split_cluster_members_df
        # Add the split-specific cluster_dict to the dictionary
        split_clusterid_to_seqid_mapping[split_name] = split_cluster_members
    return full_cluster_splits, split_clusterid_to_seqid_mapping


def read_cluster_tsv(cluster_tsv_filepath: pathlib.Path) -> Dict[str, List[str]]:
    """
    Read the cluster TSV file that is output from mmseqs2 and construct a dictionary mapping cluster representatives to sequence IDs.

    Args:
        cluster_tsv_filepath (pathlib.Path): The path to the cluster TSV file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping cluster representatives to lists of sequence IDs.
    """
    cluster_dict = {}
    with open(cluster_tsv_filepath, "r") as file:
        for line in file:
            cluster_name, sequence_name = line.strip().split("\t")
            cluster_dict.setdefault(cluster_name, []).append(sequence_name)
    return cluster_dict


def setup_clustering_file_paths(
    data_dir: str,
    file_identifier: str,
    split_sequence_similarity: float,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Set up file paths for the fasta file, cluster file, and cluster TSV file.

    Args:
        data_dir (str): The directory where the files will be stored.
        file_identifier (str): The identifier used to name the files.
        split_sequence_similarity (float): The sequence similarity threshold for splitting.

    Returns:
        Tuple[pathlib.Path, pathlib.Path, pathlib.Path]: A tuple containing the file paths for
            the input fasta file, cluster file, and cluster TSV file.
    """
    input_fasta_filepath = pathlib.Path(data_dir) / f"seq_{file_identifier}.fasta"
    cluster_filepath = (
        pathlib.Path(data_dir)
        / f"cluster_seqid_{split_sequence_similarity}_{file_identifier}_test.fasta"
    )
    cluster_tsv_filepath = cluster_filepath.with_suffix(".tsv")
    return input_fasta_filepath, cluster_filepath, cluster_tsv_filepath


def df_to_fasta(df: pd.DataFrame, output_file: str) -> None:
    """
    Convert a pandas DataFrame to a FASTA file.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'sequence' columns.
        output_file (str): Path to the output FASTA file.

    Returns:
        None
    """
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")


def fasta_to_df(fasta_input_file: str, use_modin: bool = False) -> pd.DataFrame:
    """
    Convert a FASTA file to a pandas DataFrame.

    Args:
        fasta_input_file (str): Path to the input FASTA file.
        use_modin (bool): Whether to use Modin DataFrame or pandas DataFrame. Defaults to False (pandas).

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'sequence' columns.
    """
    data = []
    with open(fasta_input_file, "r") as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    data.append([sequence_id, "".join(sequence)])
                sequence_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            data.append([sequence_id, "".join(sequence)])

        df = pd.DataFrame(data, columns=["id", "sequence"])
    return df
