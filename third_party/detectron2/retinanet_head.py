""" This is a collections of layers which expand the incoming feature layers
into box regressions and class probabilities. """

import copy
import collections
import math
from typing import Tuple, List

import torch


def depthwise(in_channels: int, out_channels: int):
    """ A depthwise separable linear layer. """
    return [
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
    ]


def conv3x3(in_channels: int, out_channels: int):
    """ Simple Conv2d layer. """
    return [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
    ]


class RetinaNetHead(torch.nn.Module):
    """This model head contains two components: classification and box regression.
    See the original RetinaNet paper for more details,
    https://arxiv.org/pdf/1708.02002.pdf."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        anchors_per_cell: int,
        num_convolutions: int = 4,  # Original paper proposes 4 convs
        use_dw: bool = False,
    ) -> None:
        super().__init__()

        if use_dw:
            conv = depthwise
        else:
            conv = conv3x3

        # Create the two subnets
        classification_subnet = torch.nn.ModuleList([])
        for idx in range(num_convolutions):
            classification_subnet += [
                *conv(in_channels, in_channels),
                torch.nn.ReLU(inplace=True),
            ]

        # NOTE same basic architecture between box regression and classification
        regression_subnet = copy.deepcopy(classification_subnet)

        # Here is where the two subnets diverge. The classification net expands the input
        # into (anchors_num * num_classes) filters because it predicts 'the probability
        # of object presence at each spatial postion for each of the A anchors
        classification_subnet += [*conv(in_channels, num_classes * anchors_per_cell)]

        # The regerssion expands the input into (4 * A) channels. So each x,y in the
        # feature map has (4 * A) channels where 4 represents (dx, dy, dw, dh). The
        # regressions for each component of each anchor box.
        regression_subnet += [*conv(in_channels, anchors_per_cell * 4)]

        # Initialize the model
        if not use_dw:
            for subnet in [regression_subnet, classification_subnet]:
                for layer in subnet.modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                        if layer.bias is not None:
                            torch.nn.init.constant_(layer.bias, 0)

        self.regression_subnet = torch.nn.Sequential(*regression_subnet)
        self.classification_subnet = torch.nn.Sequential(*classification_subnet)

        # Use prior in model initialization to improve classification stability.
        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.classification_subnet[-1].bias, bias_value)

    def __call__(
        self, feature_maps: collections.OrderedDict
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Applies the regression and classification subnets to each of the
        incoming feature maps."""

        bbox_regressions, classifications = [], []

        for level in feature_maps.values():
            bbox_regressions.append(self.regression_subnet(level))
            classifications.append(self.classification_subnet(level))

        return classifications, bbox_regressions
