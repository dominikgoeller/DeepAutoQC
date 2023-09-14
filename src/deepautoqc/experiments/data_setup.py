import lightning.pytorch as pl
import torch


class TestDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dummy_train = torch.randn((1000, 3, 256, 256))
        self.dummy_val = torch.randn((200, 3, 256, 256))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dummy_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dummy_val, batch_size=self.batch_size)
