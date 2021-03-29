""" Originally from: /detectron2/evaluation/pascal_voc_evaluation.py,
but heavily modified to make it not require XML, run faster, and accept
normalized box coordinates.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

from typing import List, Dict, Tuple, Union
import collections
import dataclasses

import torch
import numpy as np


@dataclasses.dataclass
class BoundingBox:
    box: torch.Tensor  # [x0, y0, x1, y1]
    confidence: float
    class_id: int  # _internal_ class ID


def compute_metrics(
    predictions: List[List[BoundingBox]],
    ground_truth: List[List[BoundingBox]],
    class_ids: List[int],
) -> Dict[str, float]:
    iou_30_metrics = collections.defaultdict(list)

    for class_id in class_ids:
        # Separately also compute AP30 so as not to include it into the overall
        # mAP computation. Also compute precision and recall.
        ap30, rec30 = voc_eval(predictions, ground_truth, class_id, iou_threshold=0.3)
        iou_30_metrics["ap30"].append(ap30)
        iou_30_metrics["rec30"].append(rec30)

    return {
        "ap30": np.mean(iou_30_metrics["ap30"]),
        "rec30": np.mean(iou_30_metrics["rec30"]),
    }


def voc_ap(recall: List[float], precision: List[float],) -> Tuple[float, float, float]:
    """Compute VOC AP given precision and recall."""
    # first append sentinel values at the end
    max_recall = recall[-1] if len(recall) > 0 else 0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, max_recall


# NOTE: Unlike in detectron, all boxes are normalized by width and height here.
def voc_eval(
    predictions: List[List[BoundingBox]],
    ground_truth: List[List[BoundingBox]],
    class_id: int,
    iou_threshold: float = 0.5,
) -> Union[Tuple[float, float], Tuple[float, float, float]]:

    assert len(ground_truth) == len(predictions)
    assert len(ground_truth) > 0

    class_recs = {}
    npos = 0
    for img_id, gt_image_boxes in enumerate(ground_truth):
        gt_boxes = [
            box.box.numpy() for box in gt_image_boxes if box.class_id == class_id
        ]
        npos += len(gt_boxes)
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        class_recs[img_id] = {
            "bbox": gt_boxes,
            "det": [False] * len(gt_boxes),
        }

    pred_box_img_ids = []
    pred_box_confidence = []
    pred_box_coords = []
    for img_id, pred_image_boxes in enumerate(predictions):
        pred_boxes = [box for box in pred_image_boxes if box.class_id == class_id]
        pred_box_img_ids += [img_id] * len(pred_boxes)
        pred_box_confidence += [box.confidence for box in pred_boxes]
        pred_box_coords += [box.box.tolist() for box in pred_boxes]

    predicted_bboxes = np.array(pred_box_coords, dtype=np.float32).reshape(-1, 4)
    pred_box_confidence = np.array(pred_box_confidence, dtype=np.float32)

    # Stable order is useful mostly for synthetic data, such as what we use in
    # the test.
    sorted_ind = np.argsort(-pred_box_confidence, kind="stable")
    predicted_bboxes = predicted_bboxes[sorted_ind, :]
    image_ids = [pred_box_img_ids[x] for x in sorted_ind]

    # Go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = predicted_bboxes[d, :].astype(float)
        max_overlap = -np.inf
        bb_gt = R["bbox"].astype(float)

        if bb_gt.size > 0:
            # Compute overlaps of a given box with all ground truth boxes
            # intersection
            ixmin = np.maximum(bb_gt[:, 0], bb[0])
            iymin = np.maximum(bb_gt[:, 1], bb[1])
            ixmax = np.minimum(bb_gt[:, 2], bb[2])
            iymax = np.minimum(bb_gt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.0)
            ih = np.maximum(iymax - iymin, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0]) * (bb[3] - bb[1])
                + (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
                - inters
            )
            uni[uni == 0] = -1  # So there is no division by 0.

            overlaps = inters / uni
            max_overlap = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if max_overlap > iou_threshold:
            if not R["det"][jmax]:
                # If not already mapped, mark as taken and assign true positive.
                tp[d] = 1.0
                R["det"][jmax] = True

            else:
                # Already taken: false positive
                fp[d] = 1.0
        else:
            # Does not overlap enough - false positive.
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / npos
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)

    return voc_ap(recall, precision)
