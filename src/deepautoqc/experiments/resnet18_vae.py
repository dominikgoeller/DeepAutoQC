from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn, optim

from deepautoqc.experiments.utils import SSIMLoss


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, : self.z_dim]
        logvar = x[:, self.z_dim :]
        return mu, logvar


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        # x = x.view(x.size(0), 3, 64, 64) generates output of this specific shape but regardless we have 64x64 output and upscale that
        return x


class VAE_Lightning(pl.LightningModule):
    def __init__(self, z_dim, lr, data_module):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)
        self.z_dim = z_dim
        self.loss_fn = SSIMLoss()
        self.lr = lr
        self.data_module = data_module

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decoder(z)
        x_reconstructed = F.interpolate(
            x_reconstructed,
            size=(x.size(2), x.size(3)),
            mode="bilinear",
            align_corners=False,
        )  # resize to match original input shape
        return x_reconstructed, mean, logvar

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

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def training_step(self, batch, batch_idx):
        # x = batch  # for testing only! REMOVE LATER
        x, _ = batch
        x_reconstructed, mean, logvar = self(x)

        # Reconstruction loss (you can also use other metrics like BCE)
        recon_loss = F.mse_loss(x_reconstructed, x)
        # recon_loss = self.loss_fn(x, x_reconstructed) # SSIM loss for images might be better

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_divergence

        self.log(
            "train_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_kl_divergence",
            kl_divergence,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, _ = batch
        x_reconstructed, mean, logvar = self(x)

        # Reconstruction loss (you can also use other metrics like BCE)
        recon_loss = F.mse_loss(x_reconstructed, x)
        # recon_loss = self.loss_fn(x, x_reconstructed) # SSIM loss for images might be better

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_divergence
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def test_vae():
    # Initialize the VAE model
    model = VAE_Lightning(z_dim=10, lr=1e-3)

    print(model)
    # Create random input tensor of shape (batch_size, channels, height, width)

    # For 256x256x3 input
    x1 = torch.randn(1, 3, 256, 256)
    reconstructed_x1, mean1, logvar1 = model(x1)
    print(
        f"For 256x256x3 input: Reconstructed shape: {reconstructed_x1.shape}, Mean shape: {mean1.shape}, Logvar shape: {logvar1.shape}"
    )

    # For 704x800x3 input
    x2 = torch.randn(1, 3, 704, 800)
    reconstructed_x2, mean2, logvar2 = model(x2)
    print(
        f"For 704x800x3 input: Reconstructed shape: {reconstructed_x2.shape}, Mean shape: {mean2.shape}, Logvar shape: {logvar2.shape}"
    )
