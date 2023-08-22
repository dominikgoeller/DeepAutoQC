import argparse
import os
from pathlib import Path
from typing import List

import hostlist
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from pythae.models import VAE, VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from torch import nn
from torch.utils.data import random_split

from deepautoqc.ae_architecture import Decoder, Encoder
from deepautoqc.data_structures import (
    BrainScan,
    VAE_BrainScanDataset,
    load_from_pickle,
)


def check_for_nan_and_inf(dataset):
    for i in range(len(dataset)):
        item = dataset[i]
        img = item.data
        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError(f"NaN or Inf found in the dataset at index {i}")


class VAE_Encoder(BaseEncoder):
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
        # Output of convolutional layers torch.Size([8, 128]) batchsize x latentdim
        self.mu_layer = nn.Linear(128, latent_dim)
        self.log_var_layer = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.net(x)
        print(x.shape)
        mu = self.mu_layer(x)
        log_var = F.softplus(self.log_var_layer(x))
        output = ModelOutput(embedding=mu, log_covariance=log_var)
        return output


class VAE_Decoder(BaseDecoder):
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

    def forward(self, z):
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 22, 25)
        reconstruction = self.net(x)
        output = ModelOutput(reconstruction=reconstruction)
        return output


def build_model(epochs):
    # gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
    config = BaseTrainerConfig(
        output_dir="./ckpts",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_epochs=epochs,
        seed=111,
        no_cuda=False,
        # world_size=int(os.environ["SLURM_NTASKS"]),
        # dist_backend="nccl",
        # rank=int(os.environ["SLURM_PROCID"]),
        # local_rank=int(os.environ["SLURM_LOCALID"]),
        # master_addr=hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0],
        # master_port=str(12345 + int(min(gpu_ids))),
    )
    encoder = VAE_Encoder(3, base_channel_size=32, latent_dim=128)
    decoder = VAE_Decoder(3, base_channel_size=32, latent_dim=128)
    model_config = VAEConfig(input_dim=(3, 704, 800), latent_dim=128)

    model = VAE(model_config=model_config, encoder=encoder, decoder=decoder)

    return model, config


def train_pipeline(model, config):
    pipeline = TrainingPipeline(model=model, training_config=config)

    return pipeline


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

    train_set = VAE_BrainScanDataset(train_data)
    eval_set = VAE_BrainScanDataset(eval_data)

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


def main():
    pl.seed_everything(111)
    args = parse_args()
    if args.data_location == "local":
        data_path = Path("/Volumes/PortableSSD/data/skullstrip_rpt_processed_usable")

    elif args.data_location == "cluster":
        data_path = Path(
            "/data/gpfs-1/users/goellerd_c/work/data/skullstrip_rpt_processed_usable"
        )

    EPOCHS = args.epochs

    train_set, eval_set = initialize_datasets(data_path=data_path)

    check_for_nan_and_inf(train_set)
    check_for_nan_and_inf(eval_set)

    model, training_config = build_model(epochs=EPOCHS)
    # torch.cuda.set_device(training_config.local_rank)
    # model = model.to(f"cuda:{training_config.local_rank}")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    trainer = BaseTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        training_config=training_config,
    )

    trainer.train()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #   model = nn.DataParallel(module=model)
    # model.to(device)
    torch.cuda.empty_cache()
    # pipeline = train_pipeline(model=model, config=config)
    # pipeline(train_data=train_set, eval_data=eval_set)


if __name__ == "__main__":
    main()
