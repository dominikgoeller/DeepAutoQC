import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from args import config
from data import SkullstripDataset, generate_train_validate_split
from models import resnet50
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils import (  # noqa: E402
    EarlyStopping,
    create_skullstrip_list,
    device_preparation,
    epoch_time,
    reproducibility,
)


def train_validate(
    model,
    train_l: DataLoader,
    val_l: DataLoader,
    criterion,
    optimizer,
    scheduler,
    earlystopper: EarlyStopping,
    device: torch.device,
    n_epochs: int = config.epochs,
):
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        train_loss, valid_loss, train_correct, valid_correct = 0.0, 0.0, 0.0, 0.0
        start_time = time.monotonic()
        model.train()
        train_bar = tqdm(train_l)
        train_total = 0
        for i, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            # zero the parameter gradients set_to_none=True for perfomance gains
            # source: https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)

            train_total += labels.size(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.size(
                0
            )  # in case length of dataset is not dividable by batchsize
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        valid_total = 0
        with torch.no_grad():

            valid_bar = tqdm(val_l)

            for inputs, labels in valid_bar:

                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                # probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_loss = np.round(train_loss / len(train_l.dataset), 6)
        valid_loss = np.round(valid_loss / len(val_l.dataset), 6)

        train_correct = np.round(train_correct / train_total, 6)
        valid_correct = np.round(valid_correct / valid_total, 6)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        print(f"------ Epoch: {epoch} ------")
        print(f"EpochTime:{epoch_mins}m {epoch_secs}s")
        print(f"Train loss: {train_loss}")
        print(f"Valid loss: {valid_loss}")
        print(f"Train acc: {train_correct}")
        print(f"Valid acc: {valid_correct}")
        best_model = {
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": valid_correct,
            "train_loss": train_loss,
            "val_loss": valid_loss,
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        earlystopper(valid_loss, best_model=best_model, epoch=epoch)
        if earlystopper.early_stop:
            print("Early Stopping!")
            break


def main(data_path: Path, which_optim: str):
    reproducibility()
    skullstrip_list = create_skullstrip_list(usable_dir=data_path)
    dataset = SkullstripDataset(skullstrips=skullstrip_list)
    train_loader, val_loader = generate_train_validate_split(
        dataset=dataset,
        batch_size=config.batch_size,
        seed=config.SEED,
        num_workers=config.num_workers,
    )
    model = resnet50()

    device, device_ids = device_preparation(n_gpus=config.n_gpus)
    model.to(device=device)
    if len(device_ids) > 1:
        model = nn.DataParallel(module=model, device_ids=device_ids)
    model.to(device=device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    criterion = nn.CrossEntropyLoss().to(device=device)
    if which_optim == "SGD":
        optimizer = optim.SGD(
            params=trainable_params,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=True,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif which_optim == "ADAM":
        optimizer = optim.Adam(
            params=trainable_params, lr=config.lr, betas=(0.9, 0.98), eps=1e-9
        )  # as proposed in "Attention is all you need" chapter 5.3 Optimizer
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)

    # scheduler!!
    earlystopping = EarlyStopping(verbose=True)
    train_validate(
        model=model,
        train_l=train_loader,
        val_l=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        earlystopper=earlystopping,
        device=device,
    )


if __name__ == "__main__":
    main(data_path=config.DATA_PATH, which_optim=config.optimizer)
