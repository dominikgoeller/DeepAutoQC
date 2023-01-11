import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from adan_pytorch import Adan
from args import config
from data import (
    SkullstripDataset,
    TestSkullstripDataset,
    generate_train_validate_split,
)
from models import resnet50
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils import (  # noqa: E402
    EarlyStopping,
    augment_data,
    build_save_path,
    create_skullstrip_list,
    device_preparation,
    epoch_time,
    load_from_pickle,
    reproducibility,
    resume_training,
)


def train_validate(
    model,
    train_l: DataLoader,
    val_l: DataLoader,
    criterion,
    optimizer,
    # scheduler,
    earlystopper: EarlyStopping,
    device: torch.device,
    # n_epochs: int = config.epochs,
    n_epochs: int,
):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
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
            # scheduler.step()

            train_loss += loss.item() * inputs.size(0)
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
                # probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 0]

                # f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())
                # print("LABELS:", labels)
                # print("PREDICTED:",predicted)
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

        train_accs.append(train_correct)
        val_accs.append(valid_correct)
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
            "train_accs": train_accs,
            "val_accs": val_accs,
        }
        earlystopper(valid_loss, best_model=best_model, epoch=epoch)
        if earlystopper.early_stop:
            print("Early Stopping!")
            break


def main(
    data_path: str,
    which_optim: str,
    resume_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    fine_tune: bool,
):
    reproducibility()
    # skullstrip_list = create_skullstrip_list(usable_dir=Path(data_path))
    # dataset = SkullstripDataset(skullstrips=skullstrip_list)
    # augmented_data = augment_data(datapoints=skullstrip_list)
    augmented_data = load_from_pickle(
        "/data/gpfs-1/users/goellerd_c/work/small_augmented_imageset"
    )
    dataset = TestSkullstripDataset(augmented_data)
    train_loader, val_loader = generate_train_validate_split(
        dataset=dataset,
        batch_size=batch_size,
        # batch_size=config.batch_size,
        seed=config.SEED,
        num_workers=config.num_workers,
    )

    model = resnet50(requires_grad=fine_tune)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if which_optim == "SGD":
        optimizer = optim.SGD(
            params=trainable_params,
            lr=lr,
            # lr=config.lr,
            momentum=config.momentum,
            nesterov=True,
            weight_decay=1e-4,
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif which_optim == "ADAM":
        optimizer = optim.Adam(
            params=trainable_params,
            lr=lr,
            # lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )  # as proposed in "Attention is all you need" chapter 5.3 Optimizer
    elif which_optim == "ADAN":
        optimizer = Adan(
            trainable_params,
            lr=1e-3,  # learning rate (can be much higher than Adam, up to 5-10x)
            betas=(
                0.02,
                0.08,
                0.01,
            ),  # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
            weight_decay=0.02,  # weight decay 0.02 is optimal per author
        )

    if resume_path is not None:
        model, optimizer = resume_training(model_filepath=resume_path, model=model)

    device, device_ids = device_preparation(n_gpus=config.n_gpus)
    model.to(device=device)
    if len(device_ids) > 1:
        model = nn.DataParallel(module=model, device_ids=device_ids)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)
    criterion = nn.CrossEntropyLoss().to(device=device)
    # build save path for checkpoint
    ckpt_path = build_save_path(optimizer=optimizer, fine_tune=fine_tune, model=model)
    earlystopping = EarlyStopping(path=ckpt_path, verbose=True)

    train_validate(
        model=model,
        train_l=train_loader,
        val_l=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
        earlystopper=earlystopping,
        device=device,
        n_epochs=epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-d",
        "--datapath",
        default=None,
        type=str,
        help="path to dataset used for training",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="ADAM",
        type=str,
        help="specify the optimizer either ADAM, ADAN, SGD",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to train our network for",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=16,
        help="batch size for data loaders",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        default=0.001,
        help="learning rate for training the model",
    )
    parser.add_argument(
        "-ft",
        "--fine-tune",
        action="store_true",
        help="whether to fine tune all layers or not",
    )
    args = parser.parse_args()
    data_path = args.datapath
    opt = args.optimizer
    resume_path = args.resume
    fine_tune = args.fine_tune
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    main(
        # data_path=config.DATA_PATH,
        data_path=data_path,
        # which_optim=config.optimizer,
        which_optim=opt,
        resume_path=resume_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        fine_tune=fine_tune,
    )
