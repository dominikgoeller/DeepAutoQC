import os
import pickle
from pathlib import Path


def find_error_files(directory):
    eof_error_files = []
    unpickling_error_files = []

    for file in Path(directory).glob("*"):
        if file.is_file():
            try:
                with open(file, "rb") as f:
                    pickle.load(f)
            except EOFError:
                eof_error_files.append(file.name)
            except pickle.UnpicklingError:
                unpickling_error_files.append(file.name)

    return eof_error_files, unpickling_error_files


if __name__ == "__main__":
    directory_path = "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    eof_error_files, unpickling_error_files = find_error_files(directory_path)

    print("EOF Error Files:")
    for file in eof_error_files:
        print(file)

    print("\nUnpickling Error Files:")
    for file in unpickling_error_files:
        print(file)
