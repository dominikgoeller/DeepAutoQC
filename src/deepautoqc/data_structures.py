from collections import namedtuple
from typing import List, Tuple

from numpy import typing as npt
from torch.utils.data import Dataset

BrainScan = namedtuple("BrainScan", "id, img, label")


class BrainScanDataset(Dataset):
    def __init__(self, brain_scan_list: List[BrainScan]):
        print(len(brain_scan_list))
        self.data: List[BrainScan] = brain_scan_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item: BrainScan = self.data[index]

        img: npt.NDArray = item.img
        label = item.label
        label_to_int = {"usable": 0, "unusable": 1}
        label = label_to_int[label]
        img: npt.NDArray = img.transpose((2, 0, 1))
        return img, label
