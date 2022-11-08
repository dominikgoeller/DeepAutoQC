import io
import json
import sys
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path

import cairosvg
import numpy as np
import torch
import torch.nn.functional as F
from args import config
from data import TestSkullstripDataset, generate_test_loader  # noqa: E402
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from utils import (  # noqa: E402
    create_skullstrip_list,
    device_preparation,
    load_model,
)


def svgRead(filename: str) -> np.ndarray:
    """Load an SVG file and return image in Numpy array"""
    # Convert SVG to PNG in memory
    png_content = cairosvg.svg2png(url=filename, output_height=870, output_width=2047)
    # Convert PNG to Numpy array
    res = np.array(
        Image.open(io.BytesIO(png_content))
    )  # has dimension of (870,2047,4) due to unknown reasons
    image_without_alpha = res[:, :, :3]  # drop alpha channel of image
    return image_without_alpha


def image_to_tensor(image: np.ndarray) -> np.ndarray:
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float() / 255
    return image


def test(model, test_l: DataLoader, device: torch.device):
    with torch.no_grad():
        for images in test_l:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)


def predict(model, svg_path: str):
    if torch.cuda.is_available():
        model.cuda()
    image = svgRead(filename=svg_path)
    tensor = image_to_tensor(image=image).unsqueeze(
        0
    )  # add 4th dimension as is expected input
    with torch.no_grad():
        output = model(tensor)
        _, preds = torch.max(output.data, 1)
        x = F.softmax(output, dim=1).detach().cpu().numpy()  # or output[0], dim=0
    x = np.round(x, 4)
    preds = preds.cpu().numpy()
    # usable = 1, unusable = 0

    if preds[0] == 1:
        classification = "usable"
        prob = x[0][1]
    elif preds[0] == 0:
        classification = "unusable"
        prob = x[0][0]
    prob = str(np.round(100 * prob, 2)) + "%"  # as float, ohne 100 multiplikation
    sub_name = svg_path.split("/")[
        -1
    ]  # expects input to be */*/../sub-102008_skull_strip_report.svg
    pred_dict = {
        "sub_name": sub_name,
        "prediction": classification,
        "probability": prob,
    }
    json_obj = json.dumps(pred_dict, indent=4)

    save_path = "./predictions/" + sub_name.split(sep=".")[0] + ".json"
    with open(save_path, "w") as outputfile:
        outputfile.write(json_obj)
    print(f".. Finished prediction check {save_path} for results!! ..")
    # print(f"Output of model: {output}")
    # print(f"Prediction: {preds[0]}")
    # print(f"Softmax: {x}")
    # print(prob)
    # print(classification)


def main(model_ckpt: Path, svg_path: str):

    model = load_model(model_filepath=model_ckpt)
    print(".. Finished loading model! ..")
    predict(model=model, svg_path=svg_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Input the path of your SVG file for prediction!", add_help=False
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    # Add back help
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        default=SUPPRESS,
        help="show this help message and exit",
    )
    required.add_argument("-i", "--input", help="Input path to svg file", required=True)
    args = parser.parse_args()
    svg_path = args.input
    main(model_ckpt=config.MODEL_CKPT, svg_path=svg_path)
