import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader, random_split

import wandb
from deepautoqc.data_structures import (
    BrainScan,
    BrainScanDataset_ResNet,
    load_from_pickle,
)
from deepautoqc.experiments.utils import GenerateCallback, SSIMLoss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class BasicBlockTranspose(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockTranspose, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=stride - 1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ConvTranspose2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    output_padding=stride - 1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetDecoder(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetDecoder, self).__init__()
        self.in_planes = 512

        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(block, 3, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*reversed(layers))

    def forward(self, x):
        out = self.layer4(x)
        out = self.layer3(out)
        out = self.layer2(out)
        out = self.layer1(out)
        return torch.sigmoid(out)  # Apply sigmoid to get values in [0,1] range


class Autoencoder(pl.LightningModule):
    def __init__(self, lr, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2])
        self.decoder = ResNetDecoder(BasicBlockTranspose, [2, 2, 2, 2])
        self.loss_fn = SSIMLoss()
        self.latent_dim = latent_dim
        self.lr = lr

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _get_reconstruction_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x, x_hat)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def initialize_datasets(data_path):
    pickle_paths = list(Path(data_path).glob("*.pkl"))
    data: List[BrainScan] = []
    for p in pickle_paths:
        datapoints: List[BrainScan] = load_from_pickle(p)
        data.extend(datapoints)
    print(f"Loaded data size: {len(data)}")

    train_size = int(0.8 * len(data))
    eval_size = len(data) - train_size

    train_data, eval_data = random_split(data, [train_size, eval_size])

    train_set = BrainScanDataset_ResNet(train_data)
    eval_set = BrainScanDataset_ResNet(eval_data)

    return train_set, eval_set


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
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to train our network for",
    )

    args = parser.parse_args()

    return args


def train_dataloader(train_set):
    return DataLoader(train_set, batch_size=8, shuffle=True, num_workers=12)


def val_dataloader(eval_set):
    return DataLoader(eval_set, batch_size=8, shuffle=False, num_workers=12)


def main():
    pl.seed_everything(111)
    args = parse_args()
    if args.data_location == "local":
        data_path = Path("/Users/Dominik/Charite/autoqc")

    elif args.data_location == "cluster":
        data_path = Path(
            "/data/gpfs-1/users/goellerd_c/work/data/skullstrip_rpt_processed_usable"
        )

    EPOCHS = args.epochs

    train_set, val_set = initialize_datasets(data_path=data_path)

    train_loader, val_loader = train_dataloader(train_set=train_set), val_dataloader(
        eval_set=val_set
    )

    run_name = f"ResNetAE_{args.data_location}_epochs_{EPOCHS}"
    wandb.init(project="ResNetAE_anomaly-detection", name=run_name)

    latent_dim = 128
    model = Autoencoder(latent_dim)

    reconstruction_callback = GenerateCallback(dataloader=val_loader, every_n_epochs=1)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        deterministic="warn",
        enable_progress_bar=True,
        max_epochs=EPOCHS,
        callbacks=[reconstruction_callback],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
