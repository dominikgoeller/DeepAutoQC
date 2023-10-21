import os
import pickle
from pathlib import Path


def count_unpickling_errors(directory):
    eof_errors = 0
    unpickling_errors = 0

    for file in Path(directory).glob("*"):
        if file.is_file():
            try:
                with open(file, "rb") as f:
                    pickle.load(f)
            except EOFError:
                eof_errors += 1
            except pickle.UnpicklingError:
                unpickling_errors += 1

    return eof_errors, unpickling_errors


if __name__ == "__main__":
    directory_path = "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/original_unpacked"
    eof_errors, unpickling_errors = count_unpickling_errors(directory_path)

    print(f"EOF Errors: {eof_errors}")
    print(f"Unpickling Errors: {unpickling_errors}")
