from typing import List, Dict, Tuple

import torch
from torchvision.ops import boxes

from third_party.detectron2 import postprocess
from third_party.detectron2 import regression


def smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none"
) -> torch.Tensor:
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self,
        thresholds: List[float] = [0.4, 0.5],
        labels: List[int] = [0, -1, 1],
        allow_low_quality_matches: bool = True,
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        assert all(label in [-1, 0, 1] for label in labels)
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: torch.Tensor):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i]
                indicates whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set
            # appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None], as_tuple=False
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        match_labels[pred_inds_to_update] = 1


def compute_losses(
    gt_classes: torch.Tensor,
    gt_anchors_deltas: torch.Tensor,
    cls_per_level: List[torch.Tensor],
    reg_per_level: List[torch.Tensor],
    num_classes: int,
) -> Dict[str, float]:
    """
    Args:
        gt_classes: Ground truth classes per image.
        gt_anchors: Ground truth boxes per image.
        cls_per_level: Predicted classes for each image per pyramid level.
        reg_per_level: Predicted box regression for each image per pyramid level.
        num_classes: Number of classes in the model.

    Returns:
        Mapping from a named loss to a scalar tensor storing the loss. Used
        during training only. The dict keys are: "loss_cls" and "loss_box_reg"
    """
    pred_class_logits, pred_anchor_deltas = postprocess.permute_to_N_HWA_K_and_concat(
        cls_per_level, reg_per_level, num_classes
    )

    gt_classes = gt_classes.flatten().long()
    gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

    pred_anchor_deltas = pred_anchor_deltas.view(-1, 4)
    pred_class_logits = pred_class_logits.view(-1, num_classes)

    valid_idxs = gt_classes >= 0
    foreground_idxs = (gt_classes >= 0) & (gt_classes != num_classes)
    num_foreground = foreground_idxs.sum()

    gt_classes_target = torch.zeros_like(pred_class_logits)
    gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

    # logits loss
    loss_cls = sigmoid_focal_loss(
        pred_class_logits[valid_idxs],
        gt_classes_target[valid_idxs],
        alpha=0.25,
        gamma=2.0,
        reduction="sum",
    ) / max(1, num_foreground)

    # regression loss
    loss_box_reg = smooth_l1_loss(
        pred_anchor_deltas[foreground_idxs],
        gt_anchors_deltas[foreground_idxs],
        beta=0.1,
        reduction="sum",
    ) / max(1, num_foreground)

    return loss_cls, loss_box_reg


def get_ground_truth(
    original_anchors: torch.Tensor,
    gt_boxes_all: List[torch.Tensor],
    gt_classes: List[torch.Tensor],
    num_classes: int,
    matcher: Matcher = Matcher(),
    regressor: regression.Regressor = regression.Regressor(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        original_anchors: A tensor of all the anchors in the model.
        gt_boxes_all: A list of the ground truth boxes.
        matcher: The matcher object used to match labels to original anchors.
        regressor: The regressor used to get the offsets between matched boxes.
        num_classes: The number of classes in the model.
    Returns:
        gt_classes (Tensor):
            An integer tensor of shape (N, R) storing ground-truth
            labels for each anchor.
            R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
            Anchors with an IoU with some target higher than the foreground threshold
            are assigned their corresponding label in the [0, K-1] range.
            Anchors whose IoU are below the background threshold are assigned
            the label "K". Anchors whose IoU are between the foreground and background
            thresholds are assigned a label "-1", i.e. ignore.
        gt_anchors_deltas (Tensor):
            Shape (N, R, 4).
            The last dimension represents ground-truth box2box transform
            targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
            The values in the tensor are meaningful only when the corresponding
            anchor is labeled as foreground.
    """
    gt_classes_out = []
    gt_anchors_deltas = []

    # Loop over the ground truth boxes and labels for each image in the batch.
    for gt_targets, gt_boxes in zip(gt_classes, gt_boxes_all):

        # See if there are any labels for this image, process them.
        if len(gt_boxes) > 0:

            # Calculate a IoU matrix which compares each original anchor to each
            # ground truth box. This will allow us to see which predictions match which
            # anchors.
            match_quality_matrix = boxes.box_iou(gt_boxes, original_anchors)
            gt_matched_idxs, anchor_labels = matcher(match_quality_matrix)

            # Get the ground truth regressions from the matched anchor to box labels.
            matched_gt_boxes = gt_boxes[gt_matched_idxs]

            gt_anchors_reg_deltas_i = regressor.get_deltas(
                original_anchors, matched_gt_boxes
            )
            gt_classes_i = gt_targets[gt_matched_idxs]

            # Anchors with label 0 are treated as background.
            gt_classes_i[anchor_labels == 0] = num_classes

            # Anchors with label -1 are ignored.
            gt_classes_i[anchor_labels == -1] = -1
        else:

            # If there are no labels, no anchor deltas and all boxes are background id.
            gt_anchors_reg_deltas_i = torch.zeros_like(original_anchors)
            gt_classes_i = torch.zeros(len(gt_anchors_reg_deltas_i)) + num_classes

        gt_classes_out.append(gt_classes_i)
        gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

    return torch.stack(gt_classes_out), torch.stack(gt_anchors_deltas)
