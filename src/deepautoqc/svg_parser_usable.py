from collections import namedtuple
import numpy as np
import cairosvg
from PIL import Image
import io
from xml.dom.minidom import parse, Node, Document, Element
import re
import pickle
import argparse
from svgpathtools import svg2paths,wsvg
import tempfile
import os

BrainScan = namedtuple("BrainScan", "id, img, label")

def get_transforms(item):
    #paths, attributes = svg2paths('./output_single_copy.svg')
    paths, attributes = svg2paths(item)
    transforms = []
    for path, attribute in zip(paths, attributes):
        start_x, start_y = path.start.real, path.start.imag
        transform_matrix = f"matrix(1 0 0 1 {-start_x} {-start_y-50})" # all images are beginning not at 0 but 100 for y thats why we move -50 to top
        transforms.append(transform_matrix)
    return transforms

def apply_transforms_to_svg_element(svg_element: Element, transforms):
    paths = svg_element.getElementsByTagName('path')
    for path, transform in zip(paths, transforms):
        path.setAttribute('transform', transform)
    
    

def svgRead(filename: str or io.BytesIO) -> np.ndarray:
    """Load an SVG file as bytestring and return image in Numpy array"""
    # Convert SVG to PNG in memory
    png_content = cairosvg.svg2png(bytestring=filename, output_height=200, output_width=200)
    # Convert PNG to Numpy array
    res = np.array(
        Image.open(io.BytesIO(png_content))
    )  # has dimension of (870,2047,4) due to unknown reasons
    image_without_alpha = res[:, :, :3]  # drop alpha channel of image
    return image_without_alpha

def create_svg_element(axes_el):
    doc = Document()
    
    svg = doc.createElement("svg")
    doc.appendChild(svg)

    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink")
    svg.setAttribute("version", "1.1")
    svg.appendChild(axes_el)

    return doc.toxml()

def placeholder(axes_el: Element):
    # Removing text
    for el in axes_el.getElementsByTagName('g'):
        for i in range(el.attributes.length):
            attr = el.attributes.item(i)
            if re.match(r"text_([1-9]|1[0-9]|2[01])$", attr.value):
                axes_el.removeChild(el)
                
    # Removing brain scan
    for el in axes_el.getElementsByTagName('image'):
        if 'image' in el.getAttribute('id'):
            parent: Element = el.parentNode
            parent.removeChild(el)
            
    # Changing style
    for el in axes_el.getElementsByTagName('path'):
        if 'PathCollection' in el.getAttribute('id'):
            new_style = "fill:#008000;stroke:#ffffff;stroke-width:.5"
            el.setAttribute('style', new_style)
        
    # Changing style in exception cases
    for el in axes_el.getElementsByTagName('g'):
        if 'PathCollection' in el.getAttribute('id'):
            paths = el.getElementsByTagName('path')
            new_style = "fill:#008000;stroke:#ffffff;stroke-width:.5"
            for path in paths:
                path.setAttribute('style', new_style)

def save_to_pickle(data, file_path):
    """
    This function saves the augmented data to a pickle file
    :param augmented_data: List of tuples (t1w, mask, new_label)
    :param file_path: str path where to save the pickle file
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def parse_svg(report_path: str):
    document = parse(report_path)
    g_elements = document.getElementsByTagName('g')
    axes_elements = [elem for elem in g_elements if 'axes' in elem.getAttribute('id') and 'axes_1'!=elem.getAttribute('id')]
    return axes_elements

def single_run(data_path: str, save_path: str):
    sub_list = []
    axes_elements = parse_svg(data_path)
    for item in axes_elements:
        placeholder(item)
        single_svg = create_svg_element(item)

        # write to temporary file since add_transforms svg2paths function expects a path
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
        temp_file.write(single_svg.encode())
        temp_file.close()

        transforms = get_transforms(temp_file.name)

        apply_transforms_to_svg_element(item, transforms)

        # another create_svg_element call for the transformed item
        item_transf = create_svg_element(item)
        #single_img = svgRead(temp_file.name)
        single_img = svgRead(item_transf)
        subject = BrainScan(0, single_img, "usable")
        sub_list.append(subject)
        os.unlink(temp_file.name)
    file_path = save_path + "test.pkl"
    save_to_pickle(sub_list, file_path=file_path)


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
    single_run(data_path=ARGS.datapath, save_path=ARGS.savepath)