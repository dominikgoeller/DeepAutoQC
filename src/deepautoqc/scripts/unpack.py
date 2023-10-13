from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

from deepautoqc.data_structures import BrainScan
from deepautoqc.utils import load_from_pickle, save_to_pickle


def unpack_single_pickle(p, unpacked_dir):
    datapoints: List[BrainScan] = load_from_pickle(p)
    for brain_scan in datapoints:
        new_file_path = unpacked_dir / f"{brain_scan.id}.pkl"
        save_to_pickle(brain_scan, new_file_path)


def unpack_pickles(path):
    pickle_paths = list(Path(path).glob("*.pkl"))
    print(f"Number of pickle files: {len(pickle_paths)}")

    # Create a directory for unpacked BrainScan objects
    unpacked_dir = Path(
        "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    )
    unpacked_dir.mkdir(exist_ok=True)

    with ProcessPoolExecutor() as executor:
        executor.map(
            unpack_single_pickle, pickle_paths, [unpacked_dir] * len(pickle_paths)
        )


unpack_pickles(
    "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/original"
)
