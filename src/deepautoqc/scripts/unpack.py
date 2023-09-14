import os
import pickle
from pathlib import Path
from typing import List

from deepautoqc.data_structures import BrainScan
from deepautoqc.utils import load_from_pickle, save_to_pickle


def unpack_pickles(path):
    pickle_paths = list(Path(path).glob("*.pkl"))

    print(len(pickle_paths))

    # Create a directory for unpacked BrainScan objects
    unpacked_dir = Path(
        "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    )
    unpacked_dir.mkdir(exist_ok=True)

    for p in pickle_paths:
        datapoints: List[BrainScan] = load_from_pickle(p)

        for brain_scan in datapoints:
            new_file_path = unpacked_dir / f"{brain_scan.id}.pkl"
            save_to_pickle(brain_scan, new_file_path)


unpack_pickles(
    "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original"
)