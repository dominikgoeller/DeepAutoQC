import logging
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from args import config
from halfpipe.file_index.bids import BIDSIndex
from models import resnet50
from nilearn.image import new_img_like
from scipy import ndimage
from torch import nn
from torch.optim import Optimizer
from torchvision.models import ResNet

# from transforms import (
#    BadScannerBrain,
#    BadSyntheticBrain,
#    GoodScannerBrain,
#    GoodSyntheticBrain,
# )


class EarlyStopping:
    """
    From: https://github.com/Bjarten/early-stopping-pytorch
    Regularization strategy as proposed by Ian Goodfellow et. al., “Deep Learning”, MIT Press, 2016
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        path: Path,
        patience=config.patience,
        verbose=False,
        delta=0,
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
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
            self.save_checkpoint(val_loss, best_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, best_model)
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
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ckpt = torch.load(
    #    model_filepath, map_location=torch.device("cpu")
    # )  # if you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU
    # if torch.cuda.is_available():
    #    ckpt = torch.load(model_filepath)
    # model = ckpt["model"]
    ckpt = torch.load(model_filepath, map_location=device)
    model = nn.DataParallel(resnet50()).to(device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def resume_training(model_filepath: Path, model):
    """Resume training from checkpoint.
    We are saving the entire model in model_filepath so that we don't need to load the weights again.
    Also the requires_grad attribute is saved.

    Args:
        model_filepath: Path to saved model and other variables.
    """
    if torch.cuda.is_available():
        ckpt = torch.load(model_filepath)
    ckpt = torch.load(
        model_filepath, map_location=torch.device("cpu")
    )  # if you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU
    model = ckpt["model"]
    # model = nn.DataParallel(
    #    resnet50()
    # )  # wrap resnet50 model with nn.DataParallel to not get missing_keys error!
    optimizer = ckpt["optimizer"]
    # model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return model, optimizer


def build_save_path(optimizer: Optimizer, fine_tune: bool, model: ResNet = ResNet):
    """Create folder of current date if not already exists and save modelweights to it

    Args:
        optimizer: Optimizer used in training
        model: Model used in training
    """
    # if config.requires_grad:
    #    tag = "trainable"
    # elif not config.requires_grad:
    #    tag = "frozen"
    tag = "finetune" if fine_tune else "frozen"
    directory = Path(config.EARLYSTOP_PATH + datetime.today().strftime("%Y-%m-%d"))
    directory.mkdir(parents=True, exist_ok=True)
    # file_name = Path("ResNet" + "_" + tag + "_" + optimizer.__class__.__name__ + ".pt")
    file_name = f"{model.__class__.__name__}_{tag}_{optimizer.__class__.__name__}.pt"
    ckpt_path = os.path.join(directory, file_name)
    return ckpt_path


"""
def augment_data(
    datapoints: list[tuple[Path, Path]]
) -> list[tuple[nib.Nifti1Image, nib.Nifti1Image, int]]:
    This function applies random augmentation to the datapoints using one of the four modes -
    "scanner_bad", "syn_bad", "scanner_good", "syn_good" and returns the augmented datapoints in the format (t1w, mask, new_label)
    :param datapoints: list of tuples of Path objects (t1w, mask)
    :return: list of tuples (t1w, mask, new_label)
    reproducibility()  # either use reproducibility or save file as pickle
    modes = ["scanner_bad", "syn_bad", "scanner_good", "syn_good"]
    augmented_datapoints = []
    new_label_count = {0: 0, 1: 0}
    logging.info("Started augmentation for datapoints")
    for sample in datapoints:
        mode = random.choice(modes)
        if mode == "scanner_bad":
            brain = BadScannerBrain(t1w=sample[0], mask=sample[1])
            t1w, mask = brain.apply()
            new_label = 0
        elif mode == "syn_bad":
            brain = BadSyntheticBrain(t1w=sample[0], mask=sample[1])
            t1w, mask = brain.apply()
            new_label = 0
        elif mode == "scanner_good":
            brain = GoodScannerBrain(t1w=sample[0], mask=sample[1])
            t1w, mask = brain.apply()
            new_label = 1
        elif mode == "syn_good":
            brain = GoodSyntheticBrain(t1w=sample[0], mask=sample[1])
            t1w, mask = brain.apply()
            new_label = 1
        augmented_datapoints.append((t1w, mask, new_label))
        new_label_count[new_label] += 1
    logging.info("Completed augmentation for datapoints")
    logging.info(f"Label distribution: {new_label_count}")
    return augmented_datapoints
    """


def save_to_pickle(augmented_data: list[tuple[Path, Path, int]], file_path: str):
    """
    This function saves the augmented data to a pickle file
    :param augmented_data: List of tuples (t1w, mask, new_label)
    :param file_path: str path where to save the pickle file
    """
    with open(file_path, "wb") as file:
        pickle.dump(augmented_data, file)


def load_from_pickle(file_path: str) -> list:
    """
    This function loads the augmented data from a pickle file
    :param file_path: str path where the pickle file is stored
    :return: list of tuples (t1w, mask, new_label)
    """
    with open(file_path, "rb") as file:
        augmented_data = pickle.load(file)
    return augmented_data


def random_cut_border(mask: nib.Nifti1Image, size: int = 25):
    mask_data = np.asanyarray(mask.dataobj).astype(bool)

    # Find coordinates of border voxels with argwhere returns Indices of elements that are non-zero
    # ^ xor operator with binary dilation and mask border will be one pixel next to the original mask border
    border_coords = np.argwhere(ndimage.binary_dilation(mask_data) ^ mask_data)

    # check if there are border coordinates otherwise return None
    if border_coords.shape[0] == 0:
        return None

    # print(border_coords.shape)
    # print(border_coords.shape[0])

    # Select random coordinate at the border
    rand_coord = border_coords[np.random.randint(border_coords.shape[0])]

    # print(rand_coord)
    # print(rand_coord.shape)
    # print(rand_coord[0], rand_coord[1])
    # print(mask_data.shape[0], mask_data.shape[1])

    # Find coordinates of square centered at the selected coordinate
    # and do math.floor() division to get square (size-1 if size uneven)
    x_min, x_max = max(rand_coord[0] - size // 2, 0), min(
        rand_coord[0] + size // 2, mask_data.shape[0]
    )
    y_min, y_max = max(rand_coord[1] - size // 2, 0), min(
        rand_coord[1] + size // 2, mask_data.shape[1]
    )

    assert y_max - y_min == x_max - x_min  # check if it really is square

    # print(x_min, x_max)
    # print(y_min, y_max)

    # Set voxels in square to False
    mask_data[x_min:x_max, y_min:y_max] = False

    bad_mask = new_img_like(mask, mask_data, copy_header=True)

    return bad_mask


def random_rotate_mask(mask: nib.Nifti1Image, max_angle: int = 20):
    # angles smaller than -12 degrees or bigger than 12 degrees to always generate bad masks
    angles = [random.uniform(-max_angle, -12), random.uniform(12, max_angle)]
    # angle = random.uniform(-max_angle, max_angle)
    angle = random.choice(angles)
    # angle = 12
    # print(angle)
    # mask = skullstrip.mask
    mask_data = np.asanyarray(mask.dataobj).astype(bool)

    bad_mask_data = (
        ndimage.rotate(mask_data, angle, reshape=False, axes=(0, 1), output=float) > 0.5
    )
    if not mask_data.shape == bad_mask_data.shape:
        return None
    bad_mask = new_img_like(mask, bad_mask_data, copy_header=True)

    return bad_mask
