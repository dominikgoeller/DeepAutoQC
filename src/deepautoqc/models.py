from pathlib import Path

import torch
from args import config
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

DEFAULT_WEIGHTS = Path(
    "/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/weights/resnet50-DEFAULT.pth"
)

cwd = Path.cwd()
mod_path = Path(__file__).parent
relative_path = "weights/resnet50-DEFAULT.pth"
weight_path = (mod_path / relative_path).resolve()


def resnet50(requires_grad: bool = config.requires_grad, weights_path=weight_path):
    """Load ResNet50 CNN with pretrained weights.

    Args:
        requires_grad = Boolean which determines if params should be frozen or not
    """
    # model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = models.resnet50()
    # model.load_state_dict(torch.load(DEFAULT_WEIGHTS))
    model.load_state_dict(torch.load(weights_path))
    if (
        not requires_grad
    ):  # if set to false freeze params, requires_grad is set to TRUE by default upon model loading
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    # model.avgpool is already an adaptivepoolinglayer which handles input tensors of all sizes and adapts its pooling to the specified output dims
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    return model
