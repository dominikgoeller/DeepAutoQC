import pickle
from collections import namedtuple
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import torch
import torchio as tio
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from numpy import typing as npt
from pythae.data.datasets import DatasetOutput
from sklearn.model_selection import train_test_split
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset, random_split

BrainScan = namedtuple("BrainScan", "id, img, label")


class BrainScanDataset_ResNet(Dataset):
    def __init__(self, brain_scan_list: List[BrainScan]):
        print(len(brain_scan_list))
        self.data: List[BrainScan] = brain_scan_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item: BrainScan = self.data[index]

        img: npt.NDArray = item.img
        label = item.label
        label_to_int = {"usable": 0.0, "unusable": 1.0}
        label = label_to_int[label]
        img: npt.NDArray = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        return img.float(), label


class VAE_BrainScanDataset(Dataset):
    def __init__(self, brain_scan_list: List[BrainScan]):
        print(len(brain_scan_list))
        self.data: List[BrainScan] = brain_scan_list
        self.transform = tio.CropOrPad((3, 704, 800))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item: BrainScan = self.data[index]

        img: npt.NDArray = item.img
        label = item.label
        label_to_int = {"usable": 0.0, "unusable": 1.0}
        label = label_to_int[label]
        img: npt.NDArray = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        img = tio.ScalarImage(
            tensor=img[None]
        )  # adds an extra dimension for the CropOrPad function
        img = self.transform(img)
        img = img.data[0]  # removes extra dimension again

        return DatasetOutput(data=img.float())


class BrainScanDataset(Dataset):
    def __init__(self, brain_scan_list: List[BrainScan]):
        print(len(brain_scan_list))
        self.data: List[BrainScan] = brain_scan_list
        self.transform = tio.CropOrPad((3, 704, 800))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item: BrainScan = self.data[index]

        img: npt.NDArray = item.img
        label = item.label
        label_to_int = {"usable": 0.0, "unusable": 1.0}
        label = label_to_int[label]
        img: npt.NDArray = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        img = tio.ScalarImage(
            tensor=img[None]
        )  # adds an extra dimension for the CropOrPad function
        img = self.transform(img)
        img = img.data[0]  # removes extra dimension again

        return img.float(), label


class BrainScanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        usable_path: Path,
        unusable_path: Path,
        batch_size: int,
        num_workers: int,
        seed: int = 111,
    ):
        super().__init__()
        self.usable_path = usable_path
        self.unusable_path = unusable_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        pickle_paths = list(self.usable_path.glob("*.pkl"))
        if self.unusable_path:
            pickle_paths += list(self.unusable_path.glob("*.pkl"))
        data: List[BrainScan] = []
        for p in pickle_paths:
            datapoints: List[BrainScan] = load_from_pickle(p)
            data.extend(datapoints)
        print(f"Loaded data size: {len(data)}")

        ids = [scan.id for scan in data]
        imgs = [scan.img for scan in data]
        labels = [scan.label for scan in data]

        (
            train_ids,
            test_ids,
            train_imgs,
            test_imgs,
            train_labels,
            test_labels,
        ) = train_test_split(
            ids, imgs, labels, test_size=0.2, random_state=111, stratify=labels
        )

        self.train_data = [
            BrainScan(id, img, label)
            for id, img, label in zip(train_ids, train_imgs, train_labels)
        ]
        self.test_data = [
            BrainScan(id, img, label)
            for id, img, label in zip(test_ids, test_imgs, test_labels)
        ]

    def setup(self, stage=None):
        train_set = BrainScanDataset(brain_scan_list=self.train_data)
        test_set = BrainScanDataset(brain_scan_list=self.test_data)

        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_set, self.valid_set = random_split(
            train_set, [train_set_size, valid_set_size], generator=generator
        )
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            # collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            # collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            # collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class BrainScanDataModule_lazy(pl.LightningDataModule):
    def __init__(
        self,
        usable_path: Path,
        unusable_path: Path,
        batch_size: int,
        num_workers: int,
        seed: int = 111,
    ):
        super().__init__()
        self.usable_path = usable_path
        self.unusable_path = unusable_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        # Store paths to Pickle files, limit to first 10k
        self.pickle_paths = list(self.usable_path.glob("*.pkl"))
        if self.unusable_path:
            self.pickle_paths += list(self.unusable_path.glob("*.pkl"))

    def setup(self, stage=None):
        # Load the data lazily
        data: List[BrainScan] = []
        for p in self.pickle_paths:
            datapoints: List[BrainScan] = load_from_pickle(p)
            data.extend(datapoints)

        # Perform train/test split
        ids = [scan.id for scan in data]
        imgs = [scan.img for scan in data]
        labels = [scan.label for scan in data]
        (
            train_ids,
            test_ids,
            train_imgs,
            test_imgs,
            train_labels,
            test_labels,
        ) = train_test_split(
            ids, imgs, labels, test_size=0.2, random_state=self.seed, stratify=labels
        )

        self.train_data = [
            BrainScan(id, img, label)
            for id, img, label in zip(train_ids, train_imgs, train_labels)
        ]
        self.test_data = [
            BrainScan(id, img, label)
            for id, img, label in zip(test_ids, test_imgs, test_labels)
        ]

        # Initialize datasets
        train_set = BrainScanDataset(brain_scan_list=self.train_data)
        test_set = BrainScanDataset(brain_scan_list=self.test_data)

        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_set, self.valid_set = random_split(
            train_set, [train_set_size, valid_set_size], generator=generator
        )
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def load_from_pickle(file_path: str) -> list:
    """
    This function loads the augmented data from a pickle file
    :param file_path: str path where the pickle file is stored
    :return: list of tuples (t1w, mask, new_label)
    """
    with open(file_path, "rb") as file:
        augmented_data = pickle.load(file)
    return augmented_data
