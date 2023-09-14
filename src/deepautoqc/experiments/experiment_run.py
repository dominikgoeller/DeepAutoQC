import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import WandbLogger

from deepautoqc.experiments.data_setup import (  # Assume this is your data pipeline
    TestDataModule,
)
from deepautoqc.experiments.ResNet18_AE import Autoencoder
from deepautoqc.experiments.resnet18_vae import (  # Assume these are your models
    VAE_Lightning,
)


def objective(trial):
    # Define hyperparameters using the trial object
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    z_dim = trial.suggest_categorical("z_dim", [8, 16, 32, 64, 128])
    architecture_type = trial.suggest_categorical("architecture", ["AE", "VAE"])

    # Instantiate the corresponding model
    if architecture_type == "AE":
        model = Autoencoder(lr=lr, latent_dim=z_dim)  # Add other hyperparameters
    elif architecture_type == "VAE":
        model = VAE_Lightning(z_dim=z_dim, lr=lr)  # Add other hyperparameters

    # Instantiate DataModule
    dm = TestDataModule(batch_size=trial.suggest_int("batch_size", 16, 128, step=16))

    dm.setup()

    # Instantiate logger
    # wandb_logger = WandbLogger(name="My Experiment", project="your_project_name")

    # Instantiate Trainer
    trainer = pl.Trainer(
        # logger=wandb_logger,
        max_epochs=trial.suggest_int("max_epochs", 10, 100),
        accelerator="auto",
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )

    # Train
    trainer.fit(model, dm.train_dataloader())

    # For Optuna
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
