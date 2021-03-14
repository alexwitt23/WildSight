#!/usr/bin/env python3
""" Generalized object detection training script. This script will use as many gpus as
PyTorch can find. If Nvidia's Apex is available, that will be used for mixed precision
training to speed the process up. """

import argparse
import pathlib
from typing import List, Tuple
import tarfile
import tempfile
import datetime
import json
import time
import logging
import shutil
import sys
import os
import yaml

import torch
import numpy as np
from torch.nn import parallel
from torch import distributed

from wild_sight.core import detector
from wild_sight.train.detection import dataset
from wild_sight.train.detection import collate
from wild_sight.train.train_utils import ema
from wild_sight.train.train_utils import logger
from wild_sight.train.train_utils import utils
from third_party.detectron2 import losses
from third_party.detectron2 import pascal_voc

_LOG_INTERVAL = 10
_SAVE_DIR = pathlib.Path("~/runs/wild-sight").expanduser()


def train(
    local_rank: int,
    world_size: int,
    model_cfg: dict,
    train_cfg: dict,
    data_cfg: dict,
    save_dir: pathlib.Path = None,
    initial_timestamp: pathlib.Path = None,
) -> None:
    """Main training loop that will also call out to separate evaluation function.

    Args:
        local_rank: The process's rank. 0 if there is no distributed training.
        world_size: How many devices are participating in training.
        model_cfg: The model's training params.
        train_cfg: The config of training parameters.
        save_dir: Where to save the model archive.
        initial_timestamp: The saved model to start training from.
    """

    # Do some general setup. When using distributed training and Apex, the device needs
    # to be set before loading the model.
    is_main = local_rank == 0
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    if is_main:
        log = logger.Log(save_dir / "log.txt")

    # If we are using distributed training, initialize the backend through which process
    # can communicate to each other.
    if world_size > 1:
        torch.distributed.init_process_group(
            "nccl", world_size=world_size, rank=local_rank
        )
        if is_main:
            log.info(f"Using distributed training on {world_size} gpus.")

    # Load the model.
    model = detector.Detector(
        model_params=model_cfg,
        confidence=0.05,
        num_detections_per_image=20,  # TODO(alex): Make configurable?
    )
    if initial_timestamp is not None:
        model.load_state_dict(
            torch.load(initial_timestamp / "min-loss.pt", map_location="cpu")
        )
    ema_model = ema.Ema(model)
    model.to(device)
    model.train()
    if is_main:
        log.info(f"Model architecture: \n {model}")

    image_dir = pathlib.Path(data_cfg.get("data_path")).expanduser()
    train_batch_size = train_cfg.get("train_batch_size")
    train_loader, train_sampler = create_data_loader(
        train_cfg,
        image_dir,
        image_dir / "annotations_newer.json",
        model.anchors.all_anchors,
        train_batch_size,
        world_size,
        val=False,
        image_size=model_cfg.get("img_size"),
        num_classes=model_cfg.get("num_classes"),
    )
    eval_batch_size = train_cfg.get("eval_batch_size")
    eval_loader, _ = create_data_loader(
        train_cfg,
        image_dir,
        image_dir / "annotations_newer.json",
        model.anchors.all_anchors,
        eval_batch_size,
        world_size,
        val=True,
        image_size=model_cfg.get("img_size"),
        num_classes=model_cfg.get("num_classes"),
    )
    img_size = model_cfg.get("img_size")
    # Construct the optimizer and wrap with Apex if available.
    optimizer = utils.create_optimizer(train_cfg["optimizer"], model)

    # Adjust the model for distributed training. If we are using apex, wrap the model
    # with Apex's utilies, else PyTorch.
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )
        if is_main:
            log.info("Using DistributedDataParallel.")

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    eval_start_epoch = train_cfg.get("eval_start_epoch", 10)
    eval_results, ema_eval_results = {}, {}

    lr_scheduler = None
    optimizer_cfg = train_cfg.get("optimizer", {})
    lr_config = train_cfg.get("lr_schedule", {})
    if lr_config:
        # Create the learning rate scheduler.
        warm_up_percent = lr_config.get("warmup_fraction", 0)
        start_lr = float(lr_config.get("start_lr"))
        max_lr = float(lr_config.get("max_lr"))
        end_lr = float(lr_config.get("end_lr"))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=len(train_loader) * epochs,
            final_div_factor=start_lr / end_lr,
            div_factor=max_lr / start_lr,
            pct_start=warm_up_percent,
        )

    scaler = torch.cuda.amp.GradScaler()

    # Begin training. Loop over all the epochs and run through the training data, then
    # the evaluation data. Save the best weights for the various metrics we capture.
    for epoch in range(epochs):

        all_losses, clf_losses, reg_losses = [], [], []

        # Set the train loader's epoch so data will be re-shuffled.
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        previous_loss = None

        for idx, (images, gt_regressions, gt_classes) in enumerate(train_loader):

            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            gt_regressions = gt_regressions.to(device, non_blocking=True)
            gt_classes = gt_classes.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                # Forward pass through detector
                cls_per_level, reg_per_level = model(images)

                # Compute the losses
                cls_loss, reg_loss = losses.compute_losses(
                    gt_classes=gt_classes,
                    gt_anchors_deltas=gt_regressions,
                    cls_per_level=cls_per_level,
                    reg_per_level=reg_per_level,
                    num_classes=model.module.num_classes
                    if isinstance(model, parallel.DistributedDataParallel)
                    else model.num_classes,
                )

                total_loss = cls_loss + reg_loss

            clf_losses.append(cls_loss)
            reg_losses.append(reg_loss)

            if torch.isnan(total_loss):
                raise ValueError("Loss is nan.")

            if previous_loss is None:
                previous_loss = total_loss

            if world_size > 1:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            all_losses.append(total_loss.item())
            ema_model.update(model)

            if lr_scheduler is not None:
                lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            if idx % _LOG_INTERVAL == 0 and is_main:
                log.info(
                    f"Epoch: {epoch} step {idx} ({idx % len(train_loader) / len(train_loader) * 100:.0f}%), "
                    f"clf loss {sum(clf_losses) / len(clf_losses):.5}, "
                    f"reg loss {sum(reg_losses) / len(reg_losses):.5}, "
                    f"lr {lr:.5}"
                )

                if total_loss < previous_loss:
                    if isinstance(
                        model, torch.nn.parallel.distributed.DistributedDataParallel
                    ):
                        utils.save_model(model.module, save_dir / "min-loss.pt")
                    else:
                        utils.save_model(model, save_dir / "min-loss.pt")

        # Call evaluation function if past eval delay.
        if epoch >= eval_start_epoch:

            if is_main:
                log.info("Starting evaluation")

            model.eval()
            start = time.perf_counter()
            eval_results, improved_metics = eval(
                model, eval_loader, is_main, eval_results, img_size, save_dir
            )
            model.train()

            # Call for EMA model.
            ema_eval_results, ema_improved_metrics = eval(
                ema_model.ema_model,
                eval_loader,
                is_main,
                ema_eval_results,
                img_size,
                save_dir,
            )

            if is_main:
                log.info(f"Evaluation took {time.perf_counter() - start:.3f} seconds.")
                log.info(f"Improved metrics: {improved_metics}")
                log.info(f"Improved ema metrics: {improved_metics}")
                for metric in improved_metics:
                    utils.save_model(model, save_dir / f"{metric}.pt")
                for metric in ema_improved_metrics:
                    utils.save_model(model, save_dir / f"ema-{metric}.pt")

        if is_main:
            log.info(
                f"epoch={epoch}. base={eval_results}. ema_results={ema_eval_results}"
            )


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    is_main: bool,
    previous_best: dict,
    img_size: List[int],
    save_dir: pathlib.Path = None,
) -> Tuple[dict, List[str]]:
    """ Evalulate the model against the evaulation set. Save the best weights if
    specified. Use the pycocotools package for metrics.

    Args:
        model: The model to evaluate.
        eval_loader: The eval dataset loader.
        previous_best: The current best eval metrics.
        save_dir: Where to save the model weights.

    Returns:
        The updated best metrics and a list of the metrics that improved.
    """
    detections = []
    labels = []
    for images_batch, category_ids_batch, boxes_batch in eval_loader:

        # Send ground truth to BoundingBox
        for boxes, categories in zip(boxes_batch, category_ids_batch):
            image_boxes = []
            for box, category in zip(boxes, categories.squeeze(0)):
                image_boxes.append(
                    pascal_voc.BoundingBox(
                        box / torch.Tensor(img_size * 2), 1.0, category.int().item()
                    )
                )

            labels.append(image_boxes)

        if torch.cuda.is_available():
            images_batch = images_batch.cuda()

        if isinstance(model, parallel.DistributedDataParallel):
            detection_batch = model.module.get_boxes(images_batch)
        else:
            detection_batch = model.get_boxes(images_batch)
        detections.extend(detection_batch)
    labels_list = [None] * distributed.get_world_size()
    detections_list = [None] * distributed.get_world_size()
    distributed.all_gather_object(detections_list, detections)
    distributed.all_gather_object(labels_list, labels)

    if is_main:

        labels = []
        for label_group in labels_list:
            labels.extend(label_group)
        detections = []
        for detections_group in detections_list:
            detections.extend(detections_group)

        if isinstance(model, parallel.DistributedDataParallel):
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

        metrics = pascal_voc.compute_metrics(
            detections, labels, class_ids=list(range(num_classes))
        )

        # If there are the first results, set the previous to the current.
        previous_best = metrics if not previous_best else previous_best

        improved = []
        for (metric, old), new in zip(previous_best.items(), metrics.values()):
            if new > old:
                improved.append(metric)
                previous_best[metric] = new

        return previous_best, improved
    else:
        return None, None


def create_data_loader(
    train_cfg: dict,
    data_dir: pathlib.Path,
    metadata_path: pathlib.Path,
    anchors: torch.tensor,
    batch_size: int,
    world_size: int,
    val: bool,
    image_size: int,
    num_classes: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
    """Simple function to create the dataloaders for training and evaluation.

    Args:
        training_cfg: The parameters related to the training regime.
        data_dir: The directory where the images are located.
        metadata_path: The path to the COCO metadata json.
        anchors: The tensor of anchors in the model.
        batch_size: The loader's batch size.
        world_size: World size is needed to determine if a distributed sampler is needed.
        val: Wether or not this loader is for validation.
        image_size: Size of input images into the model. NOTE: we force square images.

    Returns:
        The dataloader and the loader's sampler. For _training_ we have to set the
        sampler's epoch to reshuffle.
    """

    assert data_dir.is_dir(), data_dir

    # dataset_ = dataset.AfricanWildlife(
    #    data_dir, img_width=image_size[0], img_height=image_size[1], validation=val,
    # )
    meta = pathlib.Path(
        "/media/alex/Elements/gzgc.coco/annotations/instances_train2020.json"
    )
    dataset_ = dataset.GZGC(
        pathlib.Path("/media/alex/Elements/gzgc.coco/images/train2020"),
        metadata_path=meta,
        img_width=512,
        img_height=512,
        validation=val,
    )
    # If using distributed training, use a DistributedSampler to load exclusive sets
    # of data per process.
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset_, shuffle=val)

    if val:
        collate_fn = collate.CollateVal()
    else:
        collate_fn = collate.Collate(num_classes=num_classes, original_anchors=anchors)

    loader = torch.utils.data.DataLoader(
        dataset_,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=max(torch.multiprocessing.cpu_count() // world_size, 8),
        drop_last=True,
    )
    return loader, sampler


if __name__ == "__main__":
    # Training will always be undeterministic due to async CUDA calls,
    # but this gets us a bit closer to repeatability.
    torch.cuda.random.manual_seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for RetinaNet-based detection models."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to yaml model definition.",
    )
    parser.add_argument(
        "--initial_timestamp",
        default=None,
        type=str,
        help="Model timestamp to load as a starting point.",
    )
    args = parser.parse_args()

    # Download initial timestamp.
    initial_timestamp = None
    if args.initial_timestamp is not None:
        initial_timestamp = _SAVE_DIR / args.initial_timestamp

    config_path = args.config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    config = yaml.safe_load(config_path.read_text())
    if initial_timestamp is not None:
        config["initial_timestamp"] = args.initial_timestamp
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    save_dir = _SAVE_DIR / (
        datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    (save_dir / "config.yaml").write_text(yaml.dump(config))

    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1  # GPUS or a CPU

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            (world_size, model_cfg, train_cfg, data_cfg, save_dir, initial_timestamp,),
            nprocs=world_size,
            join=True,
        )
    else:
        train(
            0, world_size, model_cfg, train_cfg, data_cfg, save_dir, initial_timestamp
        )

    # Create tar archive.
    save_archive = save_dir / f"{save_dir.name}.tar.gz"
    with tarfile.open(save_archive, mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)

    print(f"Saved model to {save_dir / save_dir.name}.tar.gz")
