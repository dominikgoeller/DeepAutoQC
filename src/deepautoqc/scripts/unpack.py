import pickle
from pathlib import Path
from typing import List

import zstandard

from deepautoqc.data_structures import BrainScan
from deepautoqc.utils import load_from_pickle, save_to_pickle


def compress_and_save(brain_scan, compressed_dir, compressor):
    # brain_scan, compressed_dir, compressor = args
    compressed_data = compressor.compress(pickle.dumps(brain_scan))
    with open(
        Path(compressed_dir).joinpath(f"{brain_scan.id}.pkl.zst"), "wb"
    ) as compressed_file:
        compressed_file.write(compressed_data)


def unpack_single_pickle(p, save_path):
    compressed_dir = Path(save_path)
    compressed_dir.mkdir(exist_ok=True)
    # datapoints: List[BrainScan] = load_from_pickle(p)
    datapoints: List[BrainScan] = p
    compressor = zstandard.ZstdCompressor()
    for brain_scan in datapoints:
        compress_and_save(brain_scan, compressed_dir, compressor)


if __name__ == "__main__":
    pass
