from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Literal, Optional

import lightning as L
from loguru import logger
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from proteinfoundation.utils.cluster_utils import ClusterSampler
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader


class BaseLightningDataModule(L.LightningDataModule, ABC):
    """Base class for all datamodules"""

    def __init__(
        self,
        batch_padding: bool = True,
        sampling_mode: Literal["random", "cluster-random", "cluster-reps"] = "random",
        transforms: Optional[List[Callable]] = None,
        pre_transforms: Optional[List[Callable]] = None,
        pre_filters: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
    ):
        """Initialising the base data module class.

        Args:
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
            pre_transforms (List[Callable]): List of transforms applied to each example before processing.
            pre_filters (List[Callable]): List of filters applied to each example before processing.
            batch_size (int, optional): Batch size used for dataloaders. Defaults to 32.
            num_workers (int, optional): Number of workers used for dataloading. Defaults to 32.
            pin_memory (bool, optional): Whether memory should be pinned. Defaults to False.
        """
        super().__init__()
        self.batch_padding = batch_padding
        self.sampling_mode = sampling_mode
        self.transform = (
            self._compose_transforms(transforms) if transforms is not None else None
        )
        self.pre_transform = (
            self._compose_transforms(pre_transforms)
            if pre_transforms is not None
            else None
        )
        self.pre_filter = (
            self._compose_filters(pre_filters) if pre_filters is not None else None
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.clusterid_to_seqid_mappings = None  # for cluster sampling

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = self.train_dataset()
        elif stage == "validation":
            self.val_ds = self.val_dataset()
        elif stage == "test":
            self.test_ds = self.test_dataset()

    def _compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        try:
            return T.Compose(list(transforms.values()))
        except Exception:
            return T.Compose(transforms)

    def _compose_filters(self, filters: Iterable[Callable]) -> T.ComposeFilters:
        try:
            return T.ComposeFilters(list(filters.values()))
        except Exception:
            return T.ComposeFilters(filters)

    @abstractmethod
    def _get_dataset(self, split: str) -> Dataset:
        """Creates a dataset given a split.

        Args:
            split (str): Split for which to get the dataset, with options "train", "val" or "test"

        Returns:
            Dataset: Dataset created for the respective split
        """
        ...

    def train_dataset(self) -> Dataset:
        return self._get_dataset("train")

    def val_dataset(self) -> Dataset:
        return self._get_dataset("val")

    def test_dataset(self) -> Dataset:
        return self._get_dataset("test")

    def _get_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        clusterid_to_seqid_mapping: Dict[str, List[str]] = None,
    ) -> DataLoader:
        """Returns the dataloader for the corresponding dataset.

        Args:
            dataset (Dataset): PyG dataset for which the dataloader will be created.
            shuffle (bool, optional): Whether the dataloader should be shuffled. Defaults to False. False when cluster_id mapping is given.
            clusterid_to_seqid_mapping (Dict[str, List[str]], optional): Maps cluster ids to sequence ids. Defaults to None.

        Returns:
            DataLoader: Dataloader to be used by model.
        """
        if self.sampling_mode is None:
            raise ValueError(
                "Sampling mode not set, should be one of 'random', 'cluster-random' or 'cluster-reps'"
            )
        if clusterid_to_seqid_mapping and self.sampling_mode != "random":
            sampler = ClusterSampler(
                dataset=dataset,
                clusterid_to_seqid_mapping=clusterid_to_seqid_mapping,
                sampling_mode=self.sampling_mode,
            )
            shuffle = False
        elif self.sampling_mode == "random":
            sampler = None
            shuffle = shuffle
        else:
            raise ValueError(
                f"Sampling mode is {self.sampling_mode}, but clusterid_to_seqid_mapping is {clusterid_to_seqid_mapping}"
            )

        dataloader_class = DensePaddingDataLoader if self.batch_padding else DataLoader

        return dataloader_class(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            self.train_ds = self.train_dataset()
        clusterid_to_seqid_mapping = (
            self.clusterid_to_seqid_mappings["train"]
            if self.clusterid_to_seqid_mappings
            else None
        )
        shuffle = True
        train_dl = self._get_dataloader(
            dataset=self.train_ds,
            shuffle=shuffle,
            clusterid_to_seqid_mapping=clusterid_to_seqid_mapping,
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            self.val_ds = self.val_dataset()
        clusterid_to_seqid_mapping = (
            self.clusterid_to_seqid_mappings["val"]
            if self.clusterid_to_seqid_mappings
            else None
        )
        shuffle = False
        logger.info(f"Length of validation set: {len(self.val_ds)}")
        val_dl = self._get_dataloader(
            dataset=self.val_ds,
            shuffle=shuffle,
            clusterid_to_seqid_mapping=clusterid_to_seqid_mapping,
        )
        return val_dl

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            self.test_ds = self.test_dataset()
        clusterid_to_seqid_mapping = (
            self.clusterid_to_seqid_mappings["test"]
            if self.clusterid_to_seqid_mappings
            else None
        )
        shuffle = False
        test_dl = self._get_dataloader(
            dataset=self.test_ds,
            shuffle=shuffle,
            clusterid_to_seqid_mapping=clusterid_to_seqid_mapping,
        )
        return test_dl