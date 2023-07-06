from pathlib import Path
from typing import Any, List, Optional

import lightning.pytorch as pl
import torch
import torch.utils.data as data
from data_structures import BrainScan, BrainScanDataset
from lightning.pytorch.utilities.types import STEP_OUTPUT
from models import TransfusionCBRCNN
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import load_from_pickle


class MRIAutoQC(pl.LightningModule):  # type: ignore
    def __init__(self, model_name, lr=1e-3):
        super().__init__()
        self.model = TransfusionCBRCNN(
            labels=["usable", "unusable"], model_name=model_name
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        return Adam(params=self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        x, y = batch
        z = self(x)
        loss = self.criterion(z, y)
        _, predicted = torch.max(z.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        x, y = batch
        z = self(x)
        loss = self.criterion(z, y)
        _, predicted = torch.max(z.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        x, y = batch
        z = self(x)
        _, predicted = torch.max(z.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
        self.log("accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return accuracy


def generate_train_test_sets(usable_path: Path, unusable_path: Path):
    pickle_paths = list(usable_path.glob("*.pkl")) + list(unusable_path.glob("*.pkl"))
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

    train_data = [
        BrainScan(id, img, label)
        for id, img, label in zip(train_ids, train_imgs, train_labels)
    ]
    test_data = [
        BrainScan(id, img, label)
        for id, img, label in zip(test_ids, test_imgs, test_labels)
    ]

    train_brain_scan_dataset = BrainScanDataset(brain_scan_list=train_data)
    test_brain_scan_dataset = BrainScanDataset(brain_scan_list=test_data)

    return train_brain_scan_dataset, test_brain_scan_dataset


path_1 = Path("/Volumes/PortableSSD/procesed_usable")
path_2 = Path("/Volumes/PortableSSD/processed_unusable")

train_set, test_set = generate_train_test_sets(usable_path=path_1, unusable_path=path_2)

train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    train_set, [train_set_size, valid_set_size], generator=seed
)

train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set, num_workers=12)
test_loader = DataLoader(test_set)

my_model = MRIAutoQC(model_name="tall")

trainer = pl.Trainer(accelerator="auto", deterministic=True, enable_progress_bar=True)
trainer.fit(
    model=my_model, train_dataloaders=train_loader, val_dataloaders=valid_loader
)
trainer.test(my_model, dataloaders=test_loader)
