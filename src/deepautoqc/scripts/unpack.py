import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import zstandard

from deepautoqc.data_structures import BrainScan
from deepautoqc.utils import load_from_pickle, save_to_pickle


def compress_and_save(brain_scan, output_dir, compressor):
    compressed_data = compressor.compress(pickle.dumps(brain_scan))
    with open(
        Path(output_dir).joinpath(f"{brain_scan.id}.pkl.zst"), "wb"
    ) as compressed_file:
        compressed_file.write(compressed_data)


def unpack_single_pickle(p):
    unpacked_dir = Path(
        "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    )
    # compressed_dir = "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked_compressed"
    # datapoints: List[BrainScan] = load_from_pickle(p)
    datapoints: List[BrainScan] = p
    # compressor = zstandard.ZstdCompressor()
    for brain_scan in datapoints:
        # compress_and_save(brain_scan, compressed_dir, compressor)

        new_file_path = unpacked_dir.joinpath(f"{brain_scan.id}.pkl")
        save_to_pickle(brain_scan, new_file_path)


def unpack_pickles(path, unpacked_dir, compressed_dir):
    pickle_paths = list(Path(path).glob("*.pkl"))
    print(f"Number of pickle files: {len(pickle_paths)}")

    unpacked_dir = Path(unpacked_dir)
    compressed_dir = Path(compressed_dir)
    unpacked_dir.mkdir(exist_ok=True)
    compressed_dir.mkdir(exist_ok=True)

    # with ProcessPoolExecutor() as executor:
    #    executor.map(
    #        unpack_single_pickle,
    #        pickle_paths,
    #        [unpacked_dir] * len(pickle_paths),
    #        [compressed_dir] * len(pickle_paths),
    #    )
    for p in pickle_paths:
        unpack_single_pickle(p, unpacked_dir, compressed_dir)


if __name__ == "__main__":
    input_dir = "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/original"
    output_dir = "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    compressed_dir = "/data/gpfs-1/users/goellerd_c/scratch/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked_compressed"

    unpack_pickles(input_dir, output_dir, compressed_dir)
