""" A class to hold the information on a model's anchor boxes. """

from typing import List, Tuple

import torch
import numpy as np


class AnchorGenerator:
    def __init__(
        self,
        img_height: int,
        img_width: int,
        pyramid_levels: List[int],
        aspect_ratios: List[float],
        sizes: List[int] = [32, 64, 128, 256, 512],
        anchor_scales: List[float] = [1.0, 1.25, 1.5],
    ) -> None:
        """This class will hold the models anchor information. Important
        items are the anchors per feature map and the entire tensor of anchors.

        Args:
            img_height: Height of the image for calculating bbox grids.
            img_width: Width of the image for claculating bbox grids.
            pyramid_levels: Which levels are part of the feature map.
            anchor_scales: The scales by which the height and width are multiplied.

        Usage:
        >>> anchor_gen = AnchorGenerator(512, 512, [3, 4, 5, 6, 7], [0.5, 1, 2])
        >>> anchor_gen.sizes
        [32, 64, 128, 256, 512]
        >>> anchor_gen.strides
        [8, 16, 32, 64, 128]
        >>> anchor_gen.grid_sizes.tolist()
        [[64.0, 64.0], [32.0, 32.0], [16.0, 16.0], [8.0, 8.0], [4.0, 4.0]]
        >>> [cell.shape[0] for cell in anchor_gen.anchors_over_all_feature_maps]
        [36864, 9216, 2304, 576, 144]
        >>> len(anchor_gen.anchors_per_cell)   # the number of feature levels
        5
        >>> set([cell.shape for cell in anchor_gen.anchors_per_cell])
        {torch.Size([9, 4])}
        """
        self.pyramid_levels = pyramid_levels
        self.img_height = img_height
        self.img_width = img_width
        self.aspect_ratios = aspect_ratios
        self.sizes = sizes
        self.strides = [2 ** level for level in pyramid_levels]

        # Generate all anchors based on the sizes, aspect ratios, and scales. We
        # generate anchors based on aspect ratio and scale, and these are common across
        # all feature levels, but we also create the anchors based on size where the
        # size is dependent upon feature level.
        self.anchors_per_cell = self._generate_cell_anchors(
            self.sizes, self.aspect_ratios, anchor_scales
        )
        # Get the number of anchors in each cell
        self.num_anchors_per_cell = len(anchor_scales) * len(self.aspect_ratios)

        # Notice the way the grid sizes correspond to the strides. We can create grids
        # for each feature map that are the size of the original input image.
        self.grid_sizes = np.array(
            [
                [np.ceil(img_height / stride), np.ceil(img_width / stride)]
                for stride in self.strides
            ]
        )
        self.anchors_over_all_feature_maps = self._grid_anchors(self.grid_sizes)
        self.all_anchors = torch.cat(self.anchors_over_all_feature_maps)

    def _generate_cell_anchors(
        self,
        anchor_sizes: List[int],
        aspect_ratios: List[int],
        anchor_scales: List[int],
    ) -> List[torch.Tensor]:
        """Generate a tensor storing anchor boxes, which are all anchor boxes of
        different sizes, aspect_ratios, and scales centered at (0, 0). We can later
        build the set of anchors for a full feature map by shifting and tiling these
        tensors (see `meth:_grid_anchors`).
        Args:
            anchor_sizes: The width/height of the anchors.
            aspect_ratios: The aspect ratios to use to generate more anchors
                (height / width)
            anchor_scales: The values by which the anchors sizes are scaled.
        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
            in XYXY format.level
        """
        # This is different from the anchor generator defined in the original Faster
        # R-CNN code or Detectron. They yield the same AP, however the old version
        # defines cell anchors in a less natural way with a shift relative to the
        # feature grid and quantization that results in slightly different sizes for
        # different aspect ratios. See also:
        # https://github.com/facebookresearch/Detectron/issues/227

        anchors: List[torch.Tensor] = []

        # Generate anchors for each anchor size, scale, and aspect ratio
        for anchor_size in anchor_sizes:
            pyramid_anchors = []
            for anchor_scale in anchor_scales:
                # The area of the box
                area = (anchor_size * anchor_scale) ** 2
                for aspect_ratio in aspect_ratios:
                    # s * s = w * h
                    # a = h / w
                    # ... some algebra ...
                    # w = sqrt(s * s / a)
                    # h = a * w
                    w = np.sqrt(area / aspect_ratio)
                    h = aspect_ratio * w
                    x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                    pyramid_anchors.append([x0, y0, x1, y1])

            anchors.append(torch.Tensor(pyramid_anchors).float())

        return anchors

    def _grid_anchors(self, grid_sizes: np.ndarray) -> List[torch.Tensor]:
        """This function casts the anchor combinations across the feature planes.
        Remember, anchors_per_cell is only the number of combinations of aspect ratios
        and scales, over each anchor size. We need to now replicate this cell over a
        grid that spans the original image size for each feature level.

        Args:
            grid_sizes: The width, height of the grids to create for each feature map.

        Returns:
            A list of the anchors for each feature level.

        Usage:
        >>> anchor_gen = AnchorGenerator(512, 512, [3, 4, 5, 6, 7], [0.5, 1, 2])
        >>> len(anchor_gen._grid_anchors(anchor_gen.grid_sizes))
        5
        """
        anchors: List[torch.Tensor] = []
        for grid_size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.anchors_per_cell
        ):
            shift_x, shift_y = self._create_grid_offsets(grid_size, stride, offset=0.5)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def _create_grid_offsets(
        self, grid_size: List[int], stride: int, offset: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to create x, y positions for the anchor offsets on the original
        image.
        Args:
            size: The height, width of the grid to create (in pixels).
            stride: The stride, or gap, between adjacent grid points.
            offset: The amount to offset the grid from (0, 0)
        Return:
            X and Y meshgrid coordinates for the grid. These values will be added to each
            anchor cell.

        Hopefully, the following usage example is illuminating. We take a 4x4 grid, the
        same size as the most low res feature level for 512x512 input, and expand this to
        the original image.
        Usage:
        >>> anchor_gen = AnchorGenerator(512, 512, [3, 4, 5, 6, 7], [0.5, 1, 2])
        >>> x, y = anchor_gen._create_grid_offsets(
        ... anchor_gen.grid_sizes[-1], anchor_gen.strides[-1])
        >>> x.int().tolist()  # Locations of the x coordinates for lowest res feature map
        [64, 192, 320, 448, 64, 192, 320, 448, 64, 192, 320, 448, 64, 192, 320, 448]
        >>> y.int().tolist()  # Locations of the y coordinates for lowest res feature map
        [64, 64, 64, 64, 192, 192, 192, 192, 320, 320, 320, 320, 448, 448, 448, 448]
        """
        grid_height, grid_width = grid_size

        assert stride * grid_width == self.img_width
        assert stride * grid_height == self.img_height

        shifts_x = torch.arange(
            offset * stride,
            grid_width * stride,
            step=stride,
            dtype=torch.float32,
            device="cpu",
        )
        shifts_y = torch.arange(
            offset * stride,
            grid_height * stride,
            step=stride,
            dtype=torch.float32,
            device="cpu",
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        return shift_x, shift_y
