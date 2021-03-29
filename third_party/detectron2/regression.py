"""Taken from detectron2/blob/master/detectron2/modeling/box_regression.py.

This class is used to get the the regression offsets between boxes and ground truth
and apply deltas to original anchors."""

import math

import torch

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Regressor:
    def __init__(self, scale_clamp: float = _DEFAULT_SCALE_CLAMP) -> None:
        self.scale_clamp = scale_clamp
        self.weights = (1.0, 1.0, 1.0, 1.0)

    def get_deltas(
        self, src_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Get box regression transformation deltas (dx, dy, dw, dh) that can be used to
        transform the src_boxes into the target_boxes.

        Args:
            src_boxes: source boxes, e.g., object proposals
            target_boxes: target of the transformation, e.g., ground-truth boxes.
        Return:
            Returns the deltas between the srcs and target boxes.

        Usage:
        >>> regressor = Regressor()
        >>> srcs = torch.Tensor([[10, 10, 20, 20]])
        >>> targets = torch.Tensor([[10, 10, 20, 20]])
        >>> regressor.get_deltas(srcs, targets)
        tensor([[0., 0., 0., 0.]])
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)

        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas: torch.Tensor, boxes: torch.Tensor):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights

        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[..., None] + ctr_x[..., None]
        pred_ctr_y = dy * heights[..., None] + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[..., 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[..., 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[..., 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
