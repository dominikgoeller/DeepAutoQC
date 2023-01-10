import random

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from ni_image_helpers import to_image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from transforms import (
    BadScannerBrain,
    BadSyntheticBrain,
    GoodScannerBrain,
    GoodSyntheticBrain,
)
from utils import reproducibility


class SkullstripDataset(Dataset):
    """Skullstrip reports Dataset"""

    def __init__(self, skullstrips: list):

        self.skullstrips = skullstrips
        self.modes = ["scanner_bad", "syn_bad", "scanner_good", "syn_good"]
        # Transforms
        # separate konfigurationsdatei für wahrscheinlichkeiten der transforms, plotting aller funktionen zum anschauen, compose function auch verwenden
        # self.unusable_transforms_dict = {
        #    tio.RandomAffine(degrees=(20), center="image"): 0.35,
        #    tio.RandomAffine(scales=(0.85, 0.85), center="image"): 0.35,
        #    tio.RandomFlip(axes="AP", flip_probability=0.5): 0.05,
        #    tio.RandomSwap(patch_size=10, num_iterations=50): 0.05,
        #    tio.RandomNoise(): 0.05,
        #    tio.RandomSpike(num_spikes=(1, 3), intensity=(0.4, 0.6)): 0.05,
        #    tio.RandomGhosting(
        #        num_ghosts=(1, 5), intensity=(0.5, 1.2), restore=0.05
        #    ): 0.05,
        #    tio.RandomMotion(degrees=20, translation=20, num_transforms=3): 0.05,
        # }

    #
    # self.unusable_transform = tio.Compose(
    #    [
    #        tio.ToCanonical(),
    #        tio.OneOf(self.unusable_transforms_dict),
    #    ]
    # )
    #
    # self.usable_transforms_dict = {
    #    tio.RandomSwap(patch_size=5, num_iterations=50): 1,
    #    tio.RandomSpike(num_spikes=(1, 3), intensity=(0.1, 0.3)): 1,
    #    tio.RandomGamma(): 1,
    #    tio.RescaleIntensity(percentiles=(2, 98)): 1,
    # }
    #
    # self.usable_transform = tio.Compose(
    #    [
    #        tio.ToCanonical(),
    #        tio.OneOf(self.usable_transforms_dict),
    #    ]
    # )
    # self.transform = [self.usable_transform, self.unusable_transform]
    # self.transform = [transforms_cfg.good_transforms, transforms_cfg.bad_transforms]
    # self.mean = (0.485, 0.456, 0.406)
    # self.std = (0.229, 0.224, 0.225)
    # self.mean = (-1.3088, -1.2221, -0.9920) calculated on 315 training samples
    # self.std = (1.0667, 1.0705, 1.0679) calculated on 315 training samples

    def __len__(self):
        return len(self.skullstrips)

    def __getitem__(self, idx):
        sample = self.skullstrips[idx]  # Tuple: sample[0] is t1w, sample[1] is mask
        # labels = {"usable": 1, "unusable": 0}
        # new_label = random.choice(list(labels.values()))
        #
        # if new_label == 1:
        #    new_t1w = self.transform[0](nib.load(sample[0]))
        # else:
        #    new_t1w = self.transform[1](nib.load(sample[0]))

        mode = random.choice(self.modes)
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

        # image: np.ndarray = to_image(t1w=new_t1w, mask_path=sample[1])
        image: np.ndarray = to_image(t1w=t1w, mask=mask)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float() / 255
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # image = normalize(image)
        return image, new_label


class TestSkullstripDataset(Dataset):
    """Skullstrip reports Dataset"""

    def __init__(self, skullstrips: list):

        self.skullstrips = skullstrips

    def __len__(self):
        return len(self.skullstrips)

    def __getitem__(self, idx):
        sample = self.skullstrips[idx]
        t1w, mask, label = sample
        image: np.ndarray = to_image(t1w=t1w, mask=mask)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float() / 255

        return image, label


def generate_test_loader(
    dataset: TestSkullstripDataset, batchsize: int, num_workers: int
) -> DataLoader:
    test_loader = DataLoader(
        dataset=dataset, batch_size=batchsize, num_workers=num_workers
    )
    return test_loader


def generate_train_validate_split(
    dataset: SkullstripDataset, batch_size: int, seed: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Split dataset into training and validation set. Since the images will be randomly augmented every time getitem is called our network won't see the same image twice.
    70% Training and 30% Vaildation. Manual seed is set to an arbitrary number so the split happens the same every time for reproducibility.

    Args:
        dataset(SkullstripDataset): The custom SkullstripDataset which is created from a given path is used.
        batch_size: number of training samples in one forward/backward pass
        seed: seed for reproducibility
        num_workers:
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.7)
    val_size = dataset_size - train_size

    lengths = [train_size, val_size]
    training_set, validation_set = random_split(
        dataset=dataset, lengths=lengths, generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        validation_set, batch_size=batch_size, pin_memory=True, num_workers=num_workers
    )

    return train_loader, val_loader
