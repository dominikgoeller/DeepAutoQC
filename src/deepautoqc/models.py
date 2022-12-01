from pathlib import Path

import torch
from args import config
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

DEFAULT_WEIGHTS = Path("..weights/resnet50-DEFAULT.pth")


def resnet50(requires_grad: bool = config.requires_grad):
    """Load ResNet50 CNN with pretrained weights.

    Args:
        requires_grad = Boolean which determines if params should be frozen or not
    """
    # model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = models.resnet50()
    model.load_state_dict(torch.load(DEFAULT_WEIGHTS))
    if not requires_grad:  # if set to false freeze params
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    return model
