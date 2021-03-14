""" A FPN similar to the one defined in Dectectron2:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/backbone/fpn.py
"""

from typing import List
import collections

import torch


def depthwise(in_channels: int, out_channels: int):
    """ A depthwise separable linear layer. """
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
    )


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    """ Simple Conv2d layer. """
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=True
    )


class FPN(torch.nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_levels: int = 5,
        use_dw: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_in = len(in_channels)
        self.num_levels = num_levels

        if use_dw:
            conv = depthwise
        else:
            conv = conv3x3

        # Construct the lateral convolutions to adapt the incoming feature maps
        # to the same channel depth.
        self.lateral_convs = torch.nn.ModuleList([])
        for channels in reversed(in_channels):
            self.lateral_convs.append(
                torch.nn.Conv2d(channels, out_channels, kernel_size=1)
            )

        # Construct a convolution per level.
        self.convs = torch.nn.ModuleList([])
        for _ in range(self.num_levels):
            self.convs.append(conv(out_channels, out_channels))

        self.downsample_convs = torch.nn.ModuleList([])
        for _ in range(self.num_levels - self.num_in):
            self.downsample_convs.append(conv(out_channels, out_channels, stride=2))

    def __call__(
        self, feature_maps: collections.OrderedDict
    ) -> collections.OrderedDict:
        """Take the input feature maps and apply the necessary convolutions to build
        out the num_levels specified."""

        # First, loop over the incoming layers and proceed as follows: from top to
        # bottom, apply lateral convolution, add with the previous layer (if there is
        # one), and then apply a convolution.
        shapes = [[32, 32], [64, 64]]
        for idx, level_idx in enumerate(reversed(feature_maps.keys())):

            # Apply the lateral convolution to previous layer.
            feature_maps[level_idx] = self.lateral_convs[idx](feature_maps[level_idx])

            # Add the interpolated layer above, if it exists.
            if level_idx < next(reversed(feature_maps.keys())):
                feature_maps[level_idx] += torch.nn.functional.interpolate(
                    feature_maps[level_idx + 1], shapes[idx - 1], mode="nearest",
                )
            feature_maps[level_idx] = self.convs[idx](feature_maps[level_idx])

        # If more feature maps are needed to be made, take the top most incoming layer
        # and create the remaining levels.
        for idx in range(self.num_levels - self.num_in):
            new_id = next(reversed(feature_maps.keys())) + 1
            new_level = self.downsample_convs[idx](feature_maps[new_id - 1])

            # Now apply the convolution
            new_level = self.convs[self.num_in + idx](new_level)

            # Apply relu to every new level but the last.
            if idx != (self.num_levels - self.num_in - 1):
                new_level = torch.nn.functional.relu(new_level, inplace=True)

            feature_maps[new_id] = new_level

        return feature_maps
