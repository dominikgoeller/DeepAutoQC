from numpy import typing as npt
from torch.utils.data import Dataset


class BrainScanDataset(Dataset):
    def __init__(self, brain_scan_list):
        print(len(brain_scan_list))
        self.data = brain_scan_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # Preprocess your image data if needed
        img: npt.NDArray = item.img
        label = item.label
        img: npt.NDArray = img.transpose((2, 0, 1))
        return img, label
