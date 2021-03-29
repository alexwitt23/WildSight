from typing import List, Tuple
import dataclasses

import torch
from torchvision import ops

from third_party.detectron2 import regression


@dataclasses.dataclass
class BoundingBox:
    box: torch.Tensor  # [x0, y0, x1, y1]
    confidence: float
    class_id: int  # _internal_ class ID


# https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py
def permute_to_N_HWA_K(retina_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K). This is
    the tensor outputted from the RetinaNet head per pyramid level.
    Usage:
    >>> permute_to_N_HWA_K(torch.randn(1, 4 * 9, 4, 4), 4).shape
    torch.Size([1, 144, 4])
    """

    assert retina_tensor.dim() == 4, retina_tensor.shape
    N, _, H, W = retina_tensor.shape
    retina_tensor = retina_tensor.view(N, -1, num_classes, H, W)
    retina_tensor = retina_tensor.permute(0, 3, 4, 1, 2)
    retina_tensor = retina_tensor.reshape(N, -1, num_classes)  # Size=(N,HWA,K)
    return retina_tensor


def permute_to_N_HWA_K_and_concat(
    box_cls: List[torch.Tensor], box_delta: List[torch.Tensor], num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rearrange the tensor layout from the network output, i.e.: list[Tensor]: #lvl
    tensors of shape (N, A x K, Hi, Wi) to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    Args:
        box_cls: The outputted classifications per level of the retinanet head.
        box_delta: The outputted box deltas from the retinanet head.
        num_classes: The number of predicted classes.
    Returns:
        The classifications and regressions concatenated into one tensor each.
    Usage:
    """
    # For each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


# https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/wrappers.py
def cat(tensors, dim=0):
    """Efficient version of torch.cat that avoids a copy if there
    is only a single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class PostProcessor:
    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int],
        all_anchors: torch.Tensor,
        regressor: regression.Regressor,
        score_threshold: float = 0.1,
        max_detections_per_image: int = 100,
        nms_threshold: float = 0.5,
        topk_candidates: int = 100,
    ) -> None:
        """This class will parse the model's class predictions and box
        regressions and return the output boxes.

        Args:
            num_classes: The number of classes predicted.
            anchors_per_level: The anchors for each feature level.
            regressor: A class that performs the regression application.
            score_threshold: The threshold detections must meet.
            max_detections_per_image: The number of boxes predicted above
                confidence.
            nms_threshold: The non-maximal suppression threshold.
            topk_candidates: The most confident box predictions _per feature
                level_. From these, the max_detections_per_image is applied.

        """
        self.num_classes = num_classes
        self.image_size = torch.Tensor(image_size * 2).float()
        self.regressor = regressor
        self.all_anchors = all_anchors
        self.score_threshold = score_threshold
        self.topk_candidates = topk_candidates
        self.nms_threshold = nms_threshold
        self.max_detections_per_image = max_detections_per_image

    def __call__(
        self,
        box_classifications: List[torch.Tensor],
        box_regressions: List[torch.Tensor],
    ) -> List[List[BoundingBox]]:
        box_classifications = [
            permute_to_N_HWA_K(box_clf, self.num_classes)
            for box_clf in box_classifications
        ]
        box_regressions = [
            permute_to_N_HWA_K(box_reg, 4) for box_reg in box_regressions
        ]
        box_classifications = torch.cat(box_classifications, dim=1)
        box_regressions = torch.cat(box_regressions, dim=1)

        return self.inference_batch(box_classifications, box_regressions)

    def inference_batch(
        self, box_cls: torch.Tensor, box_delta: torch.Tensor
    ) -> List[List[BoundingBox]]:
        """Single-image inference. Return bounding-box detection results by
        thresholding on scores and applying non-maximum suppression (NMS).
        Arguments:
            box_cls: list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta: Same shape as 'box_cls' except that K becomes 4.
        Returns:
            A list of the bounding boxes predicted on one image.
        """

        # (B, HxWxAxK)
        box_cls = box_cls.view(box_cls.shape[0], -1).sigmoid_()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk_candidates, box_delta.size(1))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = box_cls.sort(descending=True, dim=1)

        predicted_prob = predicted_prob[:, :num_topk]
        topk_idxs = topk_idxs[:, :num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = predicted_prob > self.score_threshold
        boxes_batch = []
        for idx, keeps in enumerate(keep_idxs):

            # Select the probabilities above the threshold for each image in the batch.
            predicted_prob_ = predicted_prob[idx, keeps]
            topk_idxs_ = topk_idxs[idx, keeps]

            anchor_idxs = topk_idxs_ // self.num_classes
            classes_idxs = topk_idxs_ % self.num_classes

            # Apply the predicted deltas to the original boxes and perform NMS over all
            # the images in the batch.
            predicted_boxes = self.regressor.apply_deltas(
                box_delta[idx, anchor_idxs], self.all_anchors[anchor_idxs]
            )
            keep = ops.nms(predicted_boxes, predicted_prob_, self.nms_threshold)

            boxes = []
            for box, conf, cls_idx in zip(
                predicted_boxes[keep], predicted_prob_[keep], classes_idxs[keep]
            ):
                boxes.append(
                    BoundingBox(box.cpu() / self.image_size, float(conf), int(cls_idx))
                )
            boxes_batch.append(boxes)

        return boxes_batch
