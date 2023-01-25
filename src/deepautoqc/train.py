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
    generate_test_loader,
    generate_train_loader,
    generate_train_validate_split,
)
from metrics import confusion_matrix
from models import TransfusionCBRCNN, resnet50
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils import (  # noqa: E402; augment_data,
    EarlyStopping,
    build_save_path,
    create_skullstrip_list,
    device_preparation,
    epoch_time,
    load_from_pickle,
    load_pickle_shelve,
    reproducibility,
    resume_training,
    save_to_pickle,
    split_data,
)


def evaluate_model(trained_model, test_loader, criterion, device: torch.device):

    trained_model.eval()

    truth_labels = []
    predict_labels = []

    test_loss, test_total, test_correct = 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = trained_model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 0]
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            truth_labels.extend(labels.detach().cpu().numpy())
            predict_labels.extend(predicted.detach().cpu().numpy())

    test_loss = np.round(test_loss / len(test_loader.dataset), 6)
    test_accuracy = np.round(test_correct / test_total, 6)
    conf_matrix = confusion_matrix(actual=truth_labels, predicted=predict_labels)
    roc_auc = roc_auc_score(y_true=truth_labels, y_score=predict_labels)
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": conf_matrix,
        "roc_auc": roc_auc,
    }
    for k, v in results.items():
        print(k, v)
    save_to_pickle(data=results, file_path="./predictions/evaluation.pickle")


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
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        valid_total = 0
        # actual = []
        # predicts = []
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
                # actual.extend(labels.detach().cpu().numpy())
                # predicts.extend(predicted.detach().cpu().numpy())
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_loss = np.round(train_loss / len(train_l.dataset), 6)
        valid_loss = np.round(valid_loss / len(val_l.dataset), 6)

        train_correct = np.round(train_correct / train_total, 6)
        valid_correct = np.round(valid_correct / valid_total, 6)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        # conf_matrix = confusion_matrix(actual=actual, predicted=predicts)
        # roc_auc = roc_auc_score(y_true=actual, y_score=predicts)

        train_accs.append(train_correct)
        val_accs.append(valid_correct)
        print(f"------ Epoch: {epoch} ------")
        print(f"EpochTime:{epoch_mins}m {epoch_secs}s")
        print(f"Train loss: {train_loss}")
        print(f"Valid loss: {valid_loss}")
        print(f"Valid acc: {valid_correct}")
        # print(f"ROC_AUC_SCORE: {roc_auc}")
        best_model = {
            # "model": model,
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
            # "confusion_matrix": conf_matrix,
            # "roc_auc": roc_auc,
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
    train_augdata = load_pickle_shelve(
        "/data/gpfs-1/users/goellerd_c/work/aug_data_bigx1"
    )
    train_data, valid_data = split_data(data=train_augdata)
    train_dataset = TestSkullstripDataset(train_data)
    valid_dataset = TestSkullstripDataset(valid_data)

    # train_loader, val_loader = generate_train_validate_split(
    #    dataset=dataset,
    #    batch_size=batch_size,
    # batch_size=config.batch_size,
    #    seed=config.SEED,
    #    num_workers=config.num_workers,
    # )
    train_loader = generate_train_loader(
        dataset=train_dataset, batchsize=batch_size, num_workers=config.num_workers
    )
    val_loader = generate_test_loader(
        dataset=valid_dataset, batchsize=batch_size, num_workers=config.num_workers
    )

    model = resnet50(requires_grad=fine_tune)
    # model = TransfusionCBRCNN(labels=[0, 1], model_name="tiny")
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

    test_data = load_pickle_shelve(
        "/data/gpfs-1/users/goellerd_c/work/aug_data_smallx4"
    )
    test_dataset = TestSkullstripDataset(test_data)
    test_loader = generate_test_loader(
        dataset=test_dataset,
        batchsize=config.batch_size,
        num_workers=config.num_workers,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    evaluate_model(
        trained_model=model, test_loader=test_loader, criterion=criterion, device=device
    )


def parse_args():
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

    return args


if __name__ == "__main__":
    ARGS = parse_args()

    main(
        # data_path=config.DATA_PATH,
        data_path=ARGS.datapath,
        # which_optim=config.optimizer,
        which_optim=ARGS.optimizer,
        resume_path=ARGS.resume,
        epochs=ARGS.epochs,
        batch_size=ARGS.batch_size,
        lr=ARGS.learning_rate,
        fine_tune=ARGS.fine_tune,
    )
