import pickle
from pathlib import Path

import lightning.pytorch as pl
import numpy.typing as npt
import torch
import torchio as tio
import zstandard
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SubsetRandomSampler,
    random_split,
)

from deepautoqc.utils import load_from_pickle


class TestDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dummy_train = torch.randn((1000, 3, 256, 256))
        self.dummy_val = torch.randn((200, 3, 256, 256))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dummy_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dummy_val, batch_size=self.batch_size)


class BrainScanDataset(Dataset):
    def __init__(self, data_dir, decompress=False):
        if decompress:
            self.data_paths = list(Path(data_dir).glob("*.zst"))
        else:
            self.data_paths = list(Path(data_dir).glob("*.pkl"))
        self.transform = tio.CropOrPad((3, 704, 800))
        self.decompress = decompress

        # Create a Zstandard decompressor if needed
        if self.decompress:
            self.decompressor = zstandard.ZstdDecompressor()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        if self.decompress:
            # Load and decompress the data
            with open(self.data_paths[index], "rb") as compressed_file:
                compressed_data = compressed_file.read()
                uncompressed_data = self.decompressor.decompress(compressed_data)
                item = load_from_pickle(uncompressed_data)
        else:
            item = load_from_pickle(self.data_paths[index])

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


class BrainScanDataModule(LightningDataModule):
    def __init__(self, data_dir, decompress=False, batch_size=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.decompress = decompress
        self.train_sampler = None
        self.num_samples = 20000
        self.val_sampler = None

    def setup(self):
        brainscan_dataset = BrainScanDataset(self.data_dir, decompress=self.decompress)

        train_len = int(0.8 * len(brainscan_dataset))
        val_len = len(brainscan_dataset) - train_len

        self.brainscan_train, self.brainscan_val = random_split(
            brainscan_dataset, [train_len, val_len]
        )

    def train_dataloader(self):
        self.train_sampler = SubsetRandomSampler(
            torch.randint(high=len(self.brainscan_train), size=(self.num_samples,)),
        )
        return DataLoader(
            self.brainscan_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        self.val_sampler = SubsetRandomSampler(
            torch.randint(high=len(self.brainscan_val), size=(self.num_samples,)),
        )
        return DataLoader(
            self.brainscan_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            sampler=self.val_sampler,
        )
