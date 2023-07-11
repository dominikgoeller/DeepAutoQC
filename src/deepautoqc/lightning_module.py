import pickle
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
from torch.nn.functional import pad
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


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


def load_from_pickle(file_path: str) -> list:
    """
    This function loads the augmented data from a pickle file
    :param file_path: str path where the pickle file is stored
    :return: list of tuples (t1w, mask, new_label)
    """
    with open(file_path, "rb") as file:
        augmented_data = pickle.load(file)
    return augmented_data


class MRIAutoQC(pl.LightningModule):  # type: ignore
    def __init__(self, model_name, lr=1e-3):
        super().__init__()
        self.model = TransfusionCBRCNN(
            labels=["usable", "unusable"], model_name=model_name
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        x = x.float()
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Any:
        return Adam(params=self.model.parameters(), lr=self.lr)


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
# path_3 = Path("/data/gpfs-1/users/goellerd_c/work/data")

train_set, test_set = generate_train_test_sets(usable_path=path_1, unusable_path=path_2)

train_set_size = int(len(train_set) * 0.8)
print(train_set_size)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    train_set, [train_set_size, valid_set_size], generator=seed
)

train_loader = DataLoader(train_set, batch_size=12, collate_fn=collate_fn)
valid_loader = DataLoader(valid_set, batch_size=12, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=12, collate_fn=collate_fn)

my_model = MRIAutoQC(model_name="tall")

trainer = pl.Trainer(
    accelerator="auto", deterministic=True, enable_progress_bar=True, max_epochs=3
)

trainer.fit(
    model=my_model, train_dataloaders=train_loader, val_dataloaders=valid_loader
)
trainer.test(my_model, dataloaders=test_loader)
