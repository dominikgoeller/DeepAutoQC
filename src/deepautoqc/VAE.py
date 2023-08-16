import argparse
from pathlib import Path
from typing import List

import lightning.pytorch as pl
from pythae.models import VAE, VAEConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from torch.utils.data import random_split

from deepautoqc.data_structures import (
    BrainScan,
    VAE_BrainScanDataset,
    load_from_pickle,
)


def build_model(epochs):
    config = BaseTrainerConfig(
        output_dir="./ckpts",
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=epochs,
        seed=111,
    )

    model_config = VAEConfig(input_dim=(3, 704, 800), latent_dim=384)

    model = VAE(model_config=model_config)

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

    model, config = build_model(epochs=EPOCHS)
    pipeline = train_pipeline(model=model, config=config)
    pipeline(train_data=train_set, eval_data=eval_set)


if __name__ == "__main__":
    main()
