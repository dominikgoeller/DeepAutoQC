import base64
import io
import multiprocessing
import os
import pickle
import random
import re
import webbrowser
from copy import deepcopy
from pathlib import Path
from typing import Any, Union
from xml.dom.minidom import Document, Element

import cairosvg
import numpy as np
import numpy.typing as npt
import zstandard
from PIL import Image, ImageOps

# from data_structures import BrainScan
from deepautoqc.data_structures import BrainScan
from deepautoqc.scripts.script_utils import (
    find_reports,
    get_user_input,
    parse_args,
    parse_svg,
)
from deepautoqc.scripts.unpack import unpack_single_pickle
from deepautoqc.utils import save_to_pickle


def resize_with_padding(img, expected_size=(256, 256)):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return ImageOps.expand(img, padding)


def create_svg_from_elements(elements, image_array):
    doc = Document()

    svg = doc.createElement("svg")
    doc.appendChild(svg)

    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink")
    svg.setAttribute("version", "1.1")
    svg.setAttribute("viewBox", f"0 0 {image_array.shape[1]} {image_array.shape[0]}")
    for el in elements:
        svg.appendChild(el)

    return doc.toxml()


def find_paths(axes_el: Element):
    for path in axes_el.getElementsByTagName("path"):
        new_style = "fill:#008000"
        path.setAttribute("style", new_style)
        yield path


def parse_transform(transform: str):
    transform = transform.removeprefix("matrix(")
    transform = transform.removesuffix(")")
    tokens = transform.split(" ")

    transform_matrix = np.eye(3)
    transform_matrix[[0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]] = list(map(float, tokens))

    return transform_matrix


def transform_to_svg(matrix) -> str:
    return (
        f"matrix({' '.join(map(str, matrix[[0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]]))})"
    )


def svgRead(
    filename: Union[str, io.BytesIO], image_array: npt.NDArray[np.float64]
) -> npt.NDArray[Any]:
    if isinstance(filename, Path):
        filename = str(filename)

    png_content = cairosvg.svg2png(
        bytestring=filename,
        output_height=image_array.shape[0],
        output_width=image_array.shape[1],
    )

    res = np.array(Image.open(io.BytesIO(png_content)))
    image_without_alpha = res[:, :, :3]

    return image_without_alpha


def process_image(
    image_path: str, save_path: str, ARGS, compress: bool = False
) -> None:
    axes_elements = parse_svg(image_path)
    results = []

    base_image_name = os.path.basename(image_path).split("_skull_strip_report")[0]
    dataset_dir_match = re.search(r"(ds-[\w-]+)", image_path)
    if dataset_dir_match is not None:
        dataset_name = dataset_dir_match.group()
    else:
        dataset_name = ""

    for i, item in enumerate(axes_elements):
        (image_element,) = item.getElementsByTagName("image")
        xlink_ns = "http://www.w3.org/1999/xlink"
        base64_data = image_element.getAttributeNS(xlink_ns, "href")

        image_bytes = base64.b64decode(base64_data.split(",")[-1])
        image_obj = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image_obj)

        image_x = float(image_element.getAttribute("x"))
        image_y = float(image_element.getAttribute("y"))
        image_height = float(image_element.getAttribute("height"))
        image_width = float(image_element.getAttribute("width"))

        image_transform = image_element.getAttribute("transform")
        scale_x = image_width / image_array.shape[1]
        scale_y = image_height / image_array.shape[0]

        scale_matrix = np.eye(3)
        scale_matrix[0, 0] = scale_x
        scale_matrix[1, 1] = scale_y

        translation_matrix = np.eye(3)
        translation_matrix[0, 2] = image_x
        translation_matrix[1, 2] = image_y

        combined_affine_matrix = (
            parse_transform(image_transform) @ translation_matrix @ scale_matrix
        )

        inv_affine_matrix = np.linalg.inv(combined_affine_matrix)
        path_item = deepcopy(item)
        new_paths = list()

        for path in find_paths(path_item):
            path_id = path.getAttribute("id")

            if path_id and "PathCollection" not in path_id:
                continue

            path_transform = inv_affine_matrix
            path.setAttribute("transform", transform_to_svg(path_transform))
            new_paths.append(path)

        single_svg = create_svg_from_elements(new_paths, image_array)
        single_img = svgRead(single_svg, image_array)

        combined_image = np.zeros((image_array.shape[0], image_array.shape[1], 3))
        combined_image[:, :, 0] = image_array.mean(axis=2) / 255
        combined_image[:, :, 1] = (single_img != 0).any(axis=2)

        combined_image = np.flip(
            combined_image, axis=0
        )  # When displaying with matplotlib.pyplot images were upside down

        # img_obj = Image.fromarray(
        #    (combined_image * 255).astype(np.uint8)
        # )  # only for padding and resize function

        # resized_img = resize_with_padding(
        #    img_obj, expected_size=(256, 256)
        # )  # 256x256x3 image now!
        # resized_img_array = np.asarray(resized_img)
        # resized_img_normalized = resized_img_array.astype(np.float32) / 255

        row = (i // 7) + 1
        column = (i % 7) + 1

        result_id = f"{dataset_name}_{base_image_name}_report-skull_{row}-{column}"

        result = BrainScan(id=result_id, img=combined_image, label=ARGS.label)
        results.append(result)
    if ARGS.user:
        webbrowser.open("file://" + image_path)

        user_input = get_user_input()
        if user_input:
            user_tuples = user_input.split(";")  # Split input into tuples
            parsed_tuples = [tuple(map(int, t.split(","))) for t in user_tuples]

            # Subtract 1 from positions because of 0-based indexing
            unusable_positions = [(row - 1, col - 1) for row, col in parsed_tuples]

            # Update labels of unusable images
            for i, result in enumerate(results):
                row, col = i // 7, i % 7
                if (row, col) in unusable_positions:
                    results[i] = result._replace(label="unusable")

    if compress:
        zstd_filename = f"{dataset_name}_{base_image_name}_report-skullstrip.pkl.zst"
        compressor = zstandard.ZstdCompressor()
        compressed_data = compressor.compress(
            pickle.dumps(results)
        )  # use pickle.loads for unpacking
        with open(Path(save_path).joinpath(zstd_filename), "wb") as compressed_file:
            compressed_file.write(compressed_data)
    elif compress is False:
        # pickle_filename = f"{dataset_name}_{base_image_name}_report-skullstrip.pkl"
        # pickle_path = os.path.join(save_path, pickle_filename)
        # save_to_pickle(data=results, file_path=pickle_path)
        unpack_single_pickle(
            p=results, save_path=save_path
        )  # change function name and variables at times


def worker(image_path, save_path, ARGS):
    try:
        process_image(
            image_path=image_path,
            save_path=save_path,
            ARGS=ARGS,
            compress=ARGS.compress,
        )
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return


def delete_files(directory, filenames_to_delete):
    # Delete files that match the given filenames
    path = Path(directory)
    for file in path.iterdir():
        if any(fn in file.name for fn in filenames_to_delete):
            file.unlink()


def find_reports_random(directory, report_type):
    # Create a Path object for the directory
    path = Path(directory)

    # Filter out files that contain 'label-good' and the specific report type
    good_label_files = [
        str(file)
        for file in path.iterdir()
        if "label-good" in file.name and report_type in file.name
    ]

    # Randomly select 300 file paths
    num_files_to_select = min(300, len(good_label_files))
    selected_files = random.sample(good_label_files, num_files_to_select)

    return selected_files


def main():
    ARGS = parse_args()
    print(f"Arguments: {ARGS}")

    report_type = "skull_strip"
    # report_paths = find_reports(ARGS.datapath, report_type=report_type)
    report_paths = find_reports_random(ARGS.datapath, report_type=report_type)

    print(f"Found {len(report_paths)} reports to process.")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    for path in report_paths:
        if "label-good" in os.path.basename(path):
            pool.apply_async(worker, args=(path, ARGS.savepath, ARGS))

    pool.close()
    pool.join()

    # Extract stripped file names for deletion
    stripped_file_names = [path.stem.split("_label-good")[0] for path in report_paths]

    delete_directory = Path(
        "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/ae_data/train_compr_unpacked"
    )
    delete_files(delete_directory, stripped_file_names)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
