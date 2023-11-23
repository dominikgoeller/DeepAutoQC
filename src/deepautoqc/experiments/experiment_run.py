import lightning.pytorch as pl
import optuna
import wandb
from lightning.pytorch.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback

from deepautoqc.experiments.data_setup import (
    BrainScanDataModule,
    TestDataModule,
)
from deepautoqc.experiments.ResNet18_AE import Autoencoder
from deepautoqc.experiments.resnet18_vae import VAE_Lightning
from deepautoqc.experiments.utils import GenerateCallback

"""
Tried to use the optuna framework for hyperparameter optimization but not applicable for our current progress
"""


def objective(trial):
    wandb.init(
        project="Deep-Auto-QC",
        name=f"Trial-{trial.number}",
        dir="/data/gpfs-1/users/goellerd_c/work/wandb_init",
        config={
            "learning_rate": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 64, step=16),
            "architecture": trial.suggest_categorical("architecture", ["AE", "VAE"]),
            "z_dim": trial.suggest_categorical("z_dim", [8, 16, 32, 64, 128]),
            "max_epochs": trial.suggest_int("max_epochs", 1, 15),
        },
        notes="Optuna Trial",
        tags=["optuna", "autoencoder", "VAE", "anomaly_detection"],
    )

    wandb_logger = WandbLogger()

    lr = wandb.config.learning_rate
    z_dim = wandb.config.z_dim
    batch_size = wandb.config.batch_size
    max_epochs = wandb.config.max_epochs
    architecture_type = wandb.config.architecture

    data_dir = "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    dm = BrainScanDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.setup()

    if architecture_type == "AE":
        model = Autoencoder(lr=lr, data_module=dm, latent_dim=z_dim)
    elif architecture_type == "VAE":
        model = VAE_Lightning(z_dim=z_dim, lr=lr, data_module=dm)

    reconstruct_cb = GenerateCallback(dm.val_dataloader(), every_n_epochs=5)

    # pruning_cb = PyTorchLightningPruningCallback(trial=trial, monitor="val_loss")

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[
            reconstruct_cb,
            # pruning_cb
        ],
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    with open("all_trials.txt", "w") as f:
        for i, trial in enumerate(study.trials):
            f.write(f"Trial Number: {i+1}\n")
            f.write(f"Value: {trial.value}\n")
            f.write("Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("-" * 40 + "\n")
