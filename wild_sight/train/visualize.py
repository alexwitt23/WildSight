#!/usr/bin/env python3
"""Contains logic for finding targets in images."""

import argparse
import pathlib
import time
import json
from typing import List, Tuple, Generator

import albumentations as alb
import cv2
import torch

from wild_sight.core import detector
from wild_sight.train.detection import dataset

save_dir = pathlib.Path("/tmp/imgs")
save_dir.mkdir(exist_ok=True)

model = detector.Detector(timestamp="2021-03-14T21.12.26")
model.eval()


img_dir = pathlib.Path("/media/alex/Elements/gzgc.coco/images/train2020")
augs = alb.Compose([alb.Resize(height=512, width=512), alb.Normalize()],)
with torch.no_grad():
    for idx, image in enumerate(img_dir.glob("*.jpg")):
        image_ori = cv2.imread(str(image))
        h, w = image_ori.shape[:2]
        image = augs(image=image_ori)["image"]
        image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)

        if torch.cuda.is_available():
            image = image.cuda()

        boxes = model.get_boxes(image)

        # image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        for box in boxes[0]:
            box_t = (box.box * torch.Tensor([w, h, w, h])).int().tolist()
            cv2.rectangle(
                image_ori, (box_t[0], box_t[1]), (box_t[2], box_t[3]), (0, 255, 0), 2
            )

        cv2.imwrite(str(save_dir / f"{idx}.jpg"), image_ori)

        print(idx)
