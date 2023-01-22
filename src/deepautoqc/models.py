from pathlib import Path

import torch
from args import config
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
from torchvision import models
from torchvision.models import ResNet

DEFAULT_WEIGHTS = Path(
    "/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/weights/resnet50-DEFAULT.pth"
)

cwd = Path.cwd()
mod_path = Path(__file__).parent
relative_path = "weights/resnet50-DEFAULT.pth"
weight_path = (mod_path / relative_path).resolve()


def count_parameters(model_conv: ResNet):
    return sum(p.numel() for p in model_conv.parameters() if p.requires_grad)


def resnet50(requires_grad: bool = False, weights_path=weight_path):
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
    # model.avgpool = AdaptiveAvgPool2d(output_size=(1, 1)) which handles input tensors of all sizes and adapts its pooling to the specified output dims
    model.fc = nn.Linear(num_ftrs, config.num_classes)

    print(
        f"Finished loading model with {count_parameters(model):,} trainable parameters"
    )
    return model


def _calc_output_size(input_size, kernel_size, stride, dilation=1, padding=0):
    """Calculates output size for a given convolution configuration.
    Should work with Conv and MaxPool layers.
    See formula in docs https://pytorch.org/docs/stable/nn.html#conv2d
    """
    # pylint: disable=not-callable
    if not isinstance(input_size, torch.Tensor):
        input_size = torch.tensor(input_size)

    kernel_size = torch.tensor(kernel_size)
    stride = torch.tensor(stride)
    dilation = torch.tensor(dilation)
    padding = torch.tensor(padding)

    value = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    value = value.true_divide(stride)
    value += 1

    return torch.floor(value.float()).long()


def calc_module_output_size(model, input_size):
    """Calculates output size of a model.
    Considers only Conv2d, MaxPool2d, AvgPool2d layers.
    Tested only with Sequential layers, deeper configurations may not work
    """
    last_channel_out = None

    size = input_size
    for submodule in model.modules():
        if isinstance(submodule, (nn.Conv2d, nn.MaxPool2d)):
            size = _calc_output_size(
                size,
                submodule.kernel_size,
                submodule.stride,
                dilation=submodule.dilation,
                padding=submodule.padding,
            )
        elif isinstance(submodule, nn.AvgPool2d):
            size = _calc_output_size(
                size,
                submodule.kernel_size,
                submodule.stride,
                padding=submodule.padding,
            )

        if isinstance(submodule, nn.Conv2d):
            last_channel_out = submodule.out_channels

    return last_channel_out, tuple(size.numpy().tolist())


class GlobalMaxMeanPool2d(nn.Module):
    def forward(self, x):
        # x shape: bs, n_channels, height, width

        if self.training:
            return adaptive_avg_pool2d(x, (1, 1))
        return adaptive_max_pool2d(x, (1, 1))


def get_adaptive_pooling_layer(reduction, drop=0):
    """Returns a torch layer with AdaptivePooling2d, plus dropout if needed."""
    if reduction in ("avg", "mean"):
        reduction_step = nn.AdaptiveAvgPool2d((1, 1))
    elif reduction == "max":
        reduction_step = nn.AdaptiveMaxPool2d((1, 1))
    elif reduction == "adapt":
        reduction_step = GlobalMaxMeanPool2d()
    else:
        raise Exception(f"No such reduction {reduction}")

    layers = [reduction_step, nn.Flatten()]

    if drop > 0:
        layers.append(nn.Dropout(drop))

    return nn.Sequential(*layers)


def _cbr_layer(in_ch, out_ch, kernel_size=(7, 7), stride=1, max_pool=True):
    modules = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    ]
    if max_pool:
        modules.append(nn.MaxPool2d(3, stride=2))

    return modules


_CONFIGS = {
    # 'name': kernel_size, [layer1, layer2,...],
    "tall": ((7, 7), [32, 64, 128, 256, 512]),
    "wide": ((7, 7), [64, 128, 256, 512]),
    "small": ((7, 7), [32, 64, 128, 256]),
    "tiny": ((5, 5), [64, 128, 256, 512]),
}


def _conv_config(name, in_ch=3):
    if name not in _CONFIGS:
        raise Exception(f"Transfusion CNN not found: {name}")
    kernel_size, layers_def = _CONFIGS[name]

    n_layers = len(layers_def)

    layers = []
    for idx, out_ch in enumerate(layers_def):
        is_last = idx == n_layers - 1

        modules = _cbr_layer(in_ch, out_ch, kernel_size, max_pool=not is_last)
        layers.extend(modules)

        in_ch = out_ch

    return layers


class TransfusionCBRCNN(nn.Module):
    """Based on CBR models seen on Transfusion paper.

    Paper: Transfusion: Understanding Transfer Learning for medical imaging
    """

    def __init__(
        self,
        labels,
        pretrained_cnn=None,
        n_channels=3,
        model_name="tall",
        gpool="max",
        **unused_kwargs,
    ):
        super().__init__()

        self.labels = list(labels)
        self.model_name = model_name

        self.features = nn.Sequential(
            *_conv_config(model_name, in_ch=n_channels),
        )

        # NOTE: height and width passed are dummy, only number of channels is relevant
        out_channels, _ = calc_module_output_size(self.features, (512, 512))
        self.features_size = out_channels

        self.global_pool = get_adaptive_pooling_layer(gpool)

        self.prediction = nn.Linear(out_channels, len(self.labels))

        if pretrained_cnn is not None:
            self.load_state_dict(pretrained_cnn.state_dict())

    def forward(self, x, features=False):
        # x shape: batch_size, channels, height, width

        x = self.features(x)
        # x shape: batch_size, out_channels, h2, w2

        if features:
            return x

        x = self.global_pool(x)
        # x shape: batch_size, out_channels

        x = self.prediction(x)
        # x shape: batch_size, n_labels

        return (x,)
