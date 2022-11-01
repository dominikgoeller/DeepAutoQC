import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from args import config
from halfpipe.file_index.bids import BIDSIndex
from models import resnet50
from torch import nn


class EarlyStopping:
    """
    From: https://github.com/Bjarten/early-stopping-pytorch
    Regularization strategy as proposed by Ian Goodfellow et. al., “Deep Learning”, MIT Press, 2016
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        path: Path = config.EARLYSTOP_PATH,
        patience=5,
        verbose=False,
        delta=0,
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, best_model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, best_model, epoch=epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, best_model, epoch=epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, best_model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # save_path = self.path + datetime.today().strftime("%Y-%m-%d") + ".pt"
        torch.save(best_model, self.path)
        self.val_loss_min = val_loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def reproducibility(seed: int = 42) -> None:
    """Set seed for reproducibility

    Args:
        seed(int, optional): number used as the seed. Default 42.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_skullstrip_list(usable_dir: Path) -> list:
    """Create list of corresponding t1w, mask pairs from given path.

    Args:
        usable_dir(Path): Path where *_t1w.nii.gz and *_mask.nii.gz files are located
    """
    skullstrips = []
    bids_idx = BIDSIndex()
    bids_idx.put(root=usable_dir)

    subjects = bids_idx.get_tag_values("sub")
    for subject in subjects:
        t1w = bids_idx.get(sub=subject, suffix="T1w")
        if t1w is None:
            continue
        mask = bids_idx.get(sub=subject, suffix="mask")
        if mask is None:
            continue
        (
            t1w,
        ) = t1w  # 0.Element/einziges Element aus Set bekommen, da entweder None oder Set
        (mask,) = mask
        skullstrip = (t1w, mask)
        skullstrips.append(skullstrip)
    return skullstrips


def device_preparation(n_gpus: int) -> tuple[torch.device, list[int]]:
    """Setup GPU device if available.
    Setup for DataParallel.
    DataParallel splits your data automatically and sends job orders to multiple models on several GPUs.
    After each model finishes their job, DataParallel collects and merges the results before returning it to you.

    Args:
        n_gpus: number of gpus to use from config class of args.py file
    """
    gpu_count = torch.cuda.device_count()
    if n_gpus > 0 and gpu_count == 0:
        print(
            "Warning: There is no GPU available on this machine, training will be performed on CPU!"
        )
        n_gpus = 0
    if n_gpus > gpu_count:
        print(
            f"Warning: Number of GPUs configured to use is {n_gpus}, but only {gpu_count} are available."
        )
        n_gpus = gpu_count
    device = torch.device("cuda" if n_gpus > 0 else "cpu")
    gpu_device_ids = list(range(n_gpus))
    return device, gpu_device_ids


def load_model(model_filepath: Path):
    """Load model from checkpoint and set to eval mode."""
    if torch.cuda.is_available():
        ckpt = torch.load(model_filepath)
    ckpt = torch.load(
        model_filepath, map_location=torch.device("cpu")
    )  # if you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU
    # model = ckpt["model"]
    model = nn.DataParallel(
        resnet50()
    )  # wrap resnet50 model with nn.DataParallel to not get missing_keys error!
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def resume_training(model_filepath: Path):
    if torch.cuda.is_available():
        ckpt = torch.load(model_filepath)
    ckpt = torch.load(
        model_filepath, map_location=torch.device("cpu")
    )  # if you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU
    # model = ckpt["model"]
    model = nn.DataParallel(
        resnet50()
    )  # wrap resnet50 model with nn.DataParallel to not get missing_keys error!
    optimizer = ckpt["optimizer"]
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model.train()
    return model
