import base64
import io
import os
import re
import webbrowser
from copy import deepcopy
from typing import Any, Union
from xml.dom.minidom import Document, Element

import cairosvg
import numpy as np
import numpy.typing as npt
from PIL import Image

# from data_structures import BrainScan
from deepautoqc.data_structures import BrainScan
from deepautoqc.scripts.script_utils import (
    find_reports,
    get_user_input,
    parse_args,
    parse_svg,
)
from deepautoqc.utils import save_to_pickle


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
    png_content = cairosvg.svg2png(
        bytestring=filename,
        output_height=image_array.shape[0],
        output_width=image_array.shape[1],
    )

    res = np.array(Image.open(io.BytesIO(png_content)))
    image_without_alpha = res[:, :, :3]

    return image_without_alpha


def process_image(image_path: str, save_path: str, ARGS) -> None:
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

        row = (i // 7) + 1
        column = (i % 7) + 1

        result_id = f"{dataset_name}_{base_image_name}_report-skull_{row}-{column}"

        result = BrainScan(id=result_id, img=combined_image, label="usable")
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
    pickle_filename = f"{dataset_name}_{base_image_name}_report-skull.pkl"
    pickle_path = os.path.join(save_path, pickle_filename)
    save_to_pickle(data=results, file_path=pickle_path)


def main():
    ARGS = parse_args()
    print(f"Arguments: {ARGS}")
    report_type = "skull_strip"
    report_paths = find_reports(ARGS.datapath, report_type=report_type)
    print(len(report_paths))
    for path in report_paths:
        try:
            process_image(image_path=path, save_path=ARGS.savepath, ARGS=ARGS)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue


if __name__ == "__main__":
    main()
