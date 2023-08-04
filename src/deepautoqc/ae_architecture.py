import os
from pathlib import Path
from typing import Tuple, Type

import pytorch_lightning as pl
import pytorch_ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from piqa import SSIM

from deepautoqc.data_structures import BrainScan, BrainScanDataModule

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../ckpts/autoencoder"


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn=None,
    ):
        super().__init__()
        if act_fn is None:
            act_fn = nn.GELU()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(
                num_input_channels, c_hid, kernel_size=3, padding=1, stride=2
            ),  # 704x800 => 352x400
            act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(
                c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 352x400 => 176x200
            act_fn,
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(
                2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 176x200 => 88x100
            act_fn,
            nn.Conv2d(4 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(
                4 * c_hid, 8 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 88x100 => 44x50
            act_fn,
            nn.Conv2d(8 * c_hid, 8 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(
                8 * c_hid, 16 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 44x50 => 22x25
            act_fn,
            nn.Conv2d(16 * c_hid, 16 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(16 * 22 * 25 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn=None,
    ):
        super().__init__()
        if act_fn is None:
            act_fn = nn.GELU()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 16 * 22 * 25 * c_hid), act_fn)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                16 * c_hid,
                16 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 22x25 => 44x50
            act_fn,
            nn.Conv2d(16 * c_hid, 16 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(
                16 * c_hid,
                8 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 44x50 => 88x100
            act_fn,
            nn.Conv2d(8 * c_hid, 8 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(
                8 * c_hid,
                4 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 88x100 => 176x200
            act_fn,
            nn.Conv2d(4 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(
                4 * c_hid,
                2 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 176x200 => 352x400
            act_fn,
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(
                2 * c_hid,
                num_input_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 352x400 => 704x800
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 22, 25)
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: Type[Encoder] = Encoder,
        decoder_class: Type[Decoder] = Decoder,
        num_input_channels: int = 3,
        width: int = 800,
        height: int = 704,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        z = self.encoder(x)
        return self.decoder(z)

    def _get_reconstruction_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        x = x.double()
        x_hat = x_hat.double()  # for input of SSIM
        # loss = 1 - pytorch_ssim.ssim(x, x_hat)
        loss = SSIMLoss().cuda().forward(x, x_hat)
        return loss.mean()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)


class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(
                imgs, nrow=2, normalize=True, range=(-1, 1)
            )
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step
            )


def train_skullstrips(latent_dim):
    pl.seed_everything(111)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "skullstrip_%i" % latent_dim),
        accelerator="auto",
        devices="auto",
        deterministic="warn",
        max_epochs=1,
        # callbacks=[
        #    ModelCheckpoint(save_weights_only=True, monitor='val_loss'),
        #    #GenerateCallback(get_train_images(8), every_n_epochs=10),
        #    LearningRateMonitor(logging_interval="epoch"),
        # ],
    )
    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "skullstrip_%i.ckpt" % latent_dim)
    # if os.path.isfile(pretrained_filename):
    #    print("Found pretrained model, loading...")
    #    model = Autoencoder.load_from_checkpoint(pretrained_filename)
    # else:
    model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)

    # usable_path = Path("/Volumes/PortableSSD/data/skullstrip_rpt_processed_usable")
    usable_path = Path(
        "/data/gpfs-1/users/goellerd_c/work/data/skullstrip_rpt_processed_usable"
    )

    NUM_WORKERS = 12  # UserWarning: This DataLoader will create 64 worker processes in total. Our suggested max number of worker in current system is 12

    dm = BrainScanDataModule(
        usable_path=usable_path,
        unusable_path=usable_path,
        batch_size=12,
        num_workers=NUM_WORKERS,
    )
    dm.prepare_data()
    dm.setup()
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)
    test_result = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


def main():
    model_dict = {}
    for latent_dim in [64, 128, 256, 384]:
        model_ld, result_ld = train_skullstrips(latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
    # train_skullstrips()


if __name__ == "__main__":
    main()
