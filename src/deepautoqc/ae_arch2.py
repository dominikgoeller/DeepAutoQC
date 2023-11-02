import argparse
from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from torchsummary import summary
from torchvision.transforms import transforms

from deepautoqc.experiments.data_setup import BrainScanDataModule


class Encoder_AE(nn.Module):
    def __init__(self, z_space=64):
        super(Encoder_AE, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 176 * 200, z_space)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Decoder_AE(nn.Module):
    def __init__(self, z_space=64):
        super(Decoder_AE, self).__init__()

        self.linear = nn.Linear(z_space, 128 * 176 * 200)
        self.unflatten = nn.Unflatten(1, (128, 176, 200))

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.sigmoid(self.deconv3(x))
        return x


class Autoencoder(pl.LightningModule):
    def __init__(self, data_module):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder_AE()
        self.decoder = Decoder_AE()
        self.data_module = data_module

    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def on_train_epoch_start(self) -> None:
        indices = torch.randint(
            high=len(self.data_module.brainscan_train),
            size=(self.data_module.num_samples,),
        )
        self.data_module.train_sampler.indices = indices

    def on_validation_epoch_start(self) -> None:
        indices = torch.randint(
            high=len(self.data_module.brainscan_val),
            size=(self.data_module.num_samples,),
        )
        self.data_module.val_sampler.indices = indices

    def _get_reconstruction_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        # x = x.double()
        # x_hat = x_hat.double()  # for input of SSIM
        # loss = 1 - pytorch_ssim.ssim(x, x_hat)
        # loss = SSIMLoss().cuda().forward(x, x_hat)
        loss = F.mse_loss(x, x_hat)
        return loss.mean(), x, x_hat

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _, _ = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, x, x_hat = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % 20 == 0:  # Every 10 batches
            original = transforms.ToPILImage()(x[0].cpu())
            reconstructed = transforms.ToPILImage()(x_hat[0].cpu().detach())
            wandb.log(
                {
                    "original_images": [wandb.Image(original)],
                    "reconstructed_images": [wandb.Image(reconstructed)],
                }
            )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, _, _ = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to train our network for",
    )
    parser.add_argument(
        "-bs", "--batchsize", type=int, default=64, help="Batch Size for Training"
    )

    args = parser.parse_args()

    return args


def train_model(epochs: int, batch_size: int):
    wandb.init(
        project="autoencoder_v2", dir="/data/gpfs-1/users/goellerd_c/work/wandb_init"
    )
    wandb_logger = WandbLogger(
        save_dir="/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/wandb_logs"
    )

    data_dir = "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    dm = BrainScanDataModule(data_dir=data_dir, decompress=False, batch_size=batch_size)

    model = Autoencoder(data_module=dm)

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        deterministic="warn",
        enable_progress_bar=True,
        max_epochs=epochs,
        callbacks=ModelCheckpoint(
            monitor="val_loss",
            save_weights_only=True,
            dirpath="/data/gpfs-1/users/goellerd_c/work/wandb_logs",
            filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
        ),
    )
    tuner = Tuner(trainer=trainer)
    tuner.scale_batch_size(model=model, datamodule=dm, mode="power")

    dm.prepare_data()
    dm.setup()
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


def main():
    args = parse_args()
    torch.cuda.empty_cache()
    train_model(epochs=args.epochs, batch_size=args.batchsize)


if __name__ == "__main__":
    main()
