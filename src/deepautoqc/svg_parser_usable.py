from copy import deepcopy
from collections import namedtuple
import numpy as np
import cairosvg
from PIL import Image
import io
from xml.dom.minidom import parse, Document, Element
import re
import pickle
import base64
import matplotlib.pyplot as plt
import argparse


BrainScan = namedtuple("BrainScan", "id, img, label")

def save_to_pickle(data, file_path):
    """
    This function saves the augmented data to a pickle file
    :param augmented_data: List of tuples (t1w, mask, new_label)
    :param file_path: str path where to save the pickle file
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

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


def parse_svg(report_path: str):
    document = parse(report_path)
    g_elements = document.getElementsByTagName('g')
    axes_elements = [elem for elem in g_elements if 'axes' in elem.getAttribute('id') and 'axes_1'!=elem.getAttribute('id')]
    return axes_elements


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


def svgRead(filename: str or io.BytesIO, image_array) -> np.ndarray:
    png_content = cairosvg.svg2png(
        bytestring=filename,
        output_height=image_array.shape[0],
        output_width=image_array.shape[1],
    )

    res = np.array(
        Image.open(io.BytesIO(png_content))
    )
    image_without_alpha = res[:, :, :3]

    return image_without_alpha


def process_image(image_path, save_path):
    #BrainScan = namedtuple("BrainScan", "id, img, label")

    axes_elements = parse_svg(image_path)
    results = []

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

            if path_id and 'PathCollection' not in path_id:
                continue

            path_transform = inv_affine_matrix
            path.setAttribute("transform", transform_to_svg(path_transform))
            new_paths.append(path)
        
        single_svg = create_svg_from_elements(new_paths, image_array)
        single_img = svgRead(single_svg, image_array)

        combined_image = np.zeros((image_array.shape[0], image_array.shape[1], 3))
        combined_image[:, :, 0] = image_array.mean(axis=2) / 255
        combined_image[:, :, 1] = (single_img != 0).any(axis=2)

        combined_image = np.flip(combined_image, axis=0) # When displaying with matplotlib.pyplot images were upside down

        result = BrainScan(id=i, img=combined_image, label='usable')
        results.append(result)
    file_path = save_path + 'test.pkl'
    save_to_pickle(data=results, file_path=file_path)

def parse_args():
    parser = argparse.ArgumentParser(description="SVG Parse Script")
    parser.add_argument(
        "-d",
        "--datapath",
        help="File Path to SVG",
    )
    parser.add_argument(
        "-s",
        "--savepath",
        help="Path to save pickle",
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    ARGS = parse_args()
    result = process_image(image_path=ARGS.datapath, save_path=ARGS.savepath)

