import io
from pathlib import Path

import cairosvg
import numpy as np
import torch
from args import config
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from deepautoqc.data import TestSkullstripDataset, generate_test_loader
from deepautoqc.utils import (
    create_skullstrip_list,
    device_preparation,
    load_model,
)


def svgRead(filename):
    """Load an SVG file and return image in Numpy array"""
    # Convert SVG to PNG in memory
    png_content = cairosvg.svg2png(url=filename, output_height=870, output_width=2047)
    # Convert PNG to Numpy array
    res = np.array(
        Image.open(io.BytesIO(png_content))
    )  # has dimension of (870,2047,4) due to unknown reasons
    image_without_alpha = res[:, :, :3]  # drop alpha channel of image
    return image_without_alpha


def test(model, test_l: DataLoader, device: torch.device):
    with torch.no_grad():
        for images in test_l:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)


def main(test_data_path: Path, model_ckpt: Path):
    test_skullstrips = create_skullstrip_list(usable_dir=test_data_path)
    dataset = TestSkullstripDataset(skullstrips=test_skullstrips)

    test_loader = generate_test_loader(
        dataset=dataset, batchsize=config.batch_size, num_workers=config.num_workers
    )

    model = load_model(model_filepath=model_ckpt)

    device, device_ids = device_preparation(n_gpus=config.n_gpus)
    model.to(device=device)
    if len(device_ids) > 1:
        model = nn.DataParallel(module=model, device_ids=device_ids)
    model.to(device=device)
    test(model=model, test_l=test_loader, device=device)


if __name__ == "__main__":
    main(test_data_path=config.TEST_DATA_PATH, model_ckpt=config.DUMMY_MODEL_CKPT)
