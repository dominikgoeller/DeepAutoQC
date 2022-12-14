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
from halfpipe.file_index.bids import BIDSIndex
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


def predict_reports(model, folder_path: str):
    save_path = "./predictions/" + "exclude" + ".json"
    results = list()

    for path in Path(folder_path).rglob("*strip_report.svg"):
        pred_dict = predict(model=model, svg_path=str(path))
        results.append(pred_dict)

    with open(save_path, "w") as outputfile:
        json_obj = json.dumps(results, indent=4)
        outputfile.write(json_obj)

    print(f".. Finished prediction check {save_path} for results!! ..")


def predict(model, svg_path: str):
    image = svgRead(filename=svg_path)
    tensor = image_to_tensor(image=image).unsqueeze(
        0
    )  # add 4th dimension as is expected input
    with torch.no_grad():
        output = model(tensor)
        # _, preds = torch.max(output, dim=1)
        preds = torch.argmax(output)  # class
        x = (
            F.softmax(output, dim=1).detach().cpu().numpy()
        )  # or output[0], dim=0 probability for class
        # x = F.softmax(output, dim=1)
        # conf, classes = torch.max(x, 1)
    x = np.round(x, 4)
    preds = preds.cpu().numpy()

    # usable = 1, unusable = 0
    # print("Output", output)
    # print("output data:", output.data)
    # print("PREDS:", preds)
    # print("_:", _)
    print("x:", x)
    if preds == 1:
        classification = "good"
        prob = x[0][1]
    elif preds == 0:
        classification = "bad"
        prob = x[0][0]
    # prob = str(np.round(100 * prob, 2)) + "%"  # as float, ohne 100 multiplikation
    prob = np.round(prob, 4)
    sub_name = svg_path.split("/")[
        -1
    ]  # expects input to be */*/../sub-102008_skull_strip_report.svg
    number = sub_name.split("-")[-1].split("_")[0]
    category = "skull_strip_report"
    # pred_dict = {
    #    "sub_name": sub_name,
    #    "prediction": classification,
    #    "probability": float(prob),
    # }
    pred_dict = {
        "sub_name": number,
        "type": category,
        "rating": classification,
        "confidence": float(prob),
    }
    return pred_dict
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
    # predict(model=model, svg_path=svg_path)
    predict_reports(model=model, folder_path=svg_path)


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
