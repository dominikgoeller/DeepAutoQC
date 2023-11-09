import pickle
import random

import numpy as np
import torch
from adan_pytorch import Adan


class Adan_optimizer:
    def __init__(self, trainable_params, lr, betas, weight_decay):
        self.trainable_params = trainable_params
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def create_optimizer(self):
        optimizer = Adan(
            self.trainable_params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return optimizer


def reproducibility(seed: int = 42) -> None:
    """Set seed for reproducibility

    Args:
        seed(int, optional): number used as the seed. Default 42.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_to_pickle(data, file_path):
    """
    This function saves the augmented data to a pickle file
    :param augmented_data: List of tuples (t1w, mask, new_label)
    :param file_path: str path where to save the pickle file
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_from_pickle(file_path: str) -> list:
    """
    This function loads the augmented data from a pickle file
    :param file_path: str path where the pickle file is stored
    :return: list of tuples (t1w, mask, new_label)
    """
    with open(file_path, "rb") as file:
        augmented_data = pickle.load(file)
    return augmented_data
