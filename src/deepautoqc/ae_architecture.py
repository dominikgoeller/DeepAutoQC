import argparse
import os
from pathlib import Path
from typing import Tuple, Type

import lightning.pytorch as pl
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
from lightning.pytorch.loggers import WandbLogger
from piqa import SSIM

import wandb
from deepautoqc.data_structures import (  # BrainScanDataModule,
    BrainScan,
    BrainScanDataModule_lazy,
)
from deepautoqc.experiments.data_setup import BrainScanDataModule

# Followed the tutorial of https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
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
            nn.Flatten(),  # Image grid to single feature vector maybe change to Global Avg Pooling layer to summarize feature maps about learned objects
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
        # self.example_input_array = torch.zeros(2, num_input_channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        z = self.encoder(x)
        return self.decoder(z)

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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


class GenerateCallback(Callback):
    def __init__(self, dataloader, every_n_epochs=1):
        super().__init__()
        self.dataloader = iter(dataloader)
        self.every_n_epochs = every_n_epochs

    def get_validation_images(self):
        try:
            # Get the next batch from the validation dataloader
            images, _ = next(self.dataloader)
            return images
        except StopIteration:
            # Reset the iterator if we've reached the end of the data loader
            self.dataloader = iter(self.dataloader.__iter__())
            images, _ = next(self.dataloader)
            return images

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.get_validation_images().to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            # You can use torchvision to create a grid or use any other method to format the images.
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2)

            # Convert the PyTorch tensor to a NumPy array.
            # Make sure to transpose the dimensions to match what wandb expects.
            grid_np = grid.cpu().numpy().transpose((1, 2, 0))

            # Log the image to wandb
            trainer.logger.experiment.log(
                {
                    "Reconstructions": [
                        wandb.Image(
                            grid_np, caption="Original and Reconstructed Images"
                        )
                    ]
                }
            )


def train_skullstrips(latent_dim, epochs, data_location, batchsize):
    pl.seed_everything(111)

    # NUM_WORKERS = 12  # UserWarning: This DataLoader will create 64 worker processes in total. Our suggested max number of worker in current system is 12
    data_dir = "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    dm = BrainScanDataModule(data_dir=data_dir, decompress=False, batch_size=batchsize)
    dm.prepare_data()
    dm.setup()
    wandb.init(dir="/data/gpfs-1/users/goellerd_c/work/wandb_init")
    model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
    run_name = f"AE_{latent_dim}_epochs_{epochs}"
    wandb_logger = WandbLogger(
        project="AE_anomaly_detection",
        log_model=True,
        save_dir="/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/wandb_logs",
        name=run_name,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="/data/gpfs-1/users/goellerd_c/work/wandb_logs",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        filename=f"AE_{latent_dim}-" + "{epoch}-{step}",
    )

    # generate_callback = GenerateCallback(dm.val_dataloader(), every_n_epochs=10)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "skullstrip_%i" % latent_dim),
        accelerator="auto",
        devices="auto",
        deterministic="warn",
        enable_progress_bar=True,
        max_epochs=epochs,
        callbacks=[
            checkpoint_callback,
            # generate_callback,
            #    LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=wandb_logger,
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "skullstrip_%i.ckpt" % latent_dim)
    # if os.path.isfile(pretrained_filename):
    #    print("Found pretrained model, loading...")
    #    model = Autoencoder.load_from_checkpoint(pretrained_filename)
    # else:

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    # Test best model on validation and test set
    # val_result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)
    # test_result = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    # result = {"test": test_result, "val": val_result}
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-dl",
        "--data_location",
        type=str,
        choices=["local", "cluster"],
        required=False,
        help="Choose between 'local' and 'cluster' to determine the data paths.",
    )
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


def main():
    args = parse_args()

    # model_dict = {}
    # dims = [64, 128, 256, 384]
    for latent_dim in [128]:
        with wandb.init(
            project="AE_anomaly_detection", config={"latent_dim": latent_dim}
        ):
            model_ld, result_ld = train_skullstrips(
                latent_dim, args.epochs, args.data_location, args.batchsize
            )
            wandb.log({"Results": result_ld})
            # wandb.save("model.h5")  # Make sure to save the model in your desired format
            # model_dict[latent_dim] = {"model": model_ld, "result": result_ld}


if __name__ == "__main__":
    main()
