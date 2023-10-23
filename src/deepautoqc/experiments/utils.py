import lightning.pytorch as pl
import torch
import torchvision
from piqa import SSIM

import wandb


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1.0 - super().forward(x, y)


class GenerateCallback(pl.Callback):
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
                if isinstance(reconst_imgs, tuple):
                    reconst_imgs, _, _ = reconst_imgs  # for my vae
                pl_module.train()

            print(input_imgs)
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
