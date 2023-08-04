import pickle
from collections import namedtuple
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchio as tio
import wandb
from numpy import typing as npt
from sklearn.model_selection import train_test_split
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset

BrainScan = namedtuple("BrainScan", "id, img, label")


class LogPredictionsCallback(pl.Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [wandb.Image(img.permute(1, 2, 0)) for img in x[:n]]
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], preds[:n])
            ]

            # Log images with `WandbLogger.log_image`
            trainer.logger.experiment.log(
                {
                    "sample_images": [
                        wandb.Image(img, caption=c) for img, c in zip(images, captions)
                    ]
                }
            )

            # Log predictions as a Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i.permute(1, 2, 0)), y_i.item(), y_pred.item()]
                for x_i, y_i, y_pred in zip(x[:n], y[:n], preds[:n])
            ]
            table = wandb.Table(data=data, columns=columns)
            trainer.logger.experiment.log({"sample_table": table})


def collate_fn(batch):
    # Find the maximum height and width in this batch
    max_h = max([img.shape[1] for img, _ in batch])
    max_w = max([img.shape[2] for img, _ in batch])

    padded_imgs = []
    labels = []

    for img, label in batch:
        # img = torch.from_numpy(img)
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        pad_h_up = pad_h // 2
        pad_h_down = pad_h - pad_h_up
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # Pad the image symmetrically
        padded_img = pad(img, pad=(pad_w_left, pad_w_right, pad_h_up, pad_h_down))

        padded_imgs.append(padded_img)
        labels.append(label)

    # Stack images and labels
    padded_imgs = torch.stack(padded_imgs)
    labels = torch.tensor(labels)

    return padded_imgs, labels


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
        label_to_int = {"usable": 0, "unusable": 1}
        label = label_to_int[label]
        img: npt.NDArray = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        img = tio.ScalarImage(
            tensor=img[None]
        )  # adds an extra dimension for the CropOrPad function
        img = self.transform(img)
        img = img.data[0]  # removes extra dimension again

        return img, label


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
        pickle_paths = list(self.usable_path.glob("*.pkl")) + list(
            self.unusable_path.glob("*.pkl")
        )
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
        self.train_set, self.valid_set = data.random_split(
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
            collate_fn=collate_fn,
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
