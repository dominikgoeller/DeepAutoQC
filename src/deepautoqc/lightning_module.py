import argparse
from pathlib import Path
from typing import Any, List, Optional

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim import Adam
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from deepautoqc.data_structures import (
    BrainScan,
    BrainScanDataModule,
    LogPredictionsCallback,
)
from deepautoqc.models import TransfusionCBRCNN


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
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Any:
        return Adam(params=self.model.parameters(), lr=self.lr)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-dl",
        "--data_location",
        type=str,
        choices=["local", "cluster"],
        required=True,
        help="Choose between 'local' and 'cluster' to determine the data paths.",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        choices=["small", "tiny", "wide", "tall"],
        required=True,
        help="Choose between 'small', 'tiny', 'wide', 'tall' to determine model architecture.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to train our network for",
    )

    args = parser.parse_args()

    return args


def main():
    pl.seed_everything(111)
    args = parse_args()
    if args.data_location == "local":
        usable_path = Path("/Volumes/PortableSSD/procesed_usable")
        unusable_path = Path("/Volumes/PortableSSD/processed_unusable")
    elif args.data_location == "cluster":
        usable_path = Path(
            "/data/gpfs-1/users/goellerd_c/work/data/skullstrip_rpt_processed_usable"
        )
        unusable_path = Path(
            "/data/gpfs-1/users/goellerd_c/work/data/skullstrip_rpt_processed_unusable"
        )

    NUM_WORKERS = 12  # UserWarning: This DataLoader will create 64 worker processes in total. Our suggested max number of worker in current system is 12

    dm = BrainScanDataModule(
        usable_path=usable_path,
        unusable_path=unusable_path,
        batch_size=12,
        num_workers=NUM_WORKERS,
    )
    dm.prepare_data()
    dm.setup()

    my_model = MRIAutoQC(model_name=args.model_name)

    # log_predictions_callback = LogPredictionsCallback()

    checkpoint_callback = ModelCheckpoint(
        dirpath="/data/gpfs-1/users/goellerd_c/work/wandb_logs",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        filename=f"{args.model_name}-" + "{epoch}-{step}",
    )
    wandb_logger = WandbLogger(
        project="DeepAutoQC",
        log_model=True,
        save_dir="/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/wandb_logs",
    )

    trainer = pl.Trainer(
        accelerator="auto",
        deterministic="warn",
        enable_progress_bar=True,
        max_epochs=args.epochs,
        strategy="auto",
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    trainer.fit(
        model=my_model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    trainer.test(my_model, dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()
