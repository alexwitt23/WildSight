#!/usr/bin/env python3
"""A script to convert a dataset of YOLO annotations to a COCO dataset."""

import pathlib
import json
import shutil

from PIL import Image
import cv2


_SAVE_DIR = pathlib.Path("/home/alex/datasets/swift-parrots/dataset-withoutails")
_IMG_SAVE_DIR = _SAVE_DIR / "images"
_IMG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
_IMG_DIR = pathlib.Path("/home/alex/datasets/swift-parrots/CURRENT")
_LABELS_DIR = pathlib.Path("/home/alex/datasets/swift-parrots/LABELS_WITHOUT_TAILS")
_IMG_EXTS = [".jpg", ".jpeg", ".png"]


coco_dataset = {
    "categories": [{"name": "parrot", "id": 0}],
    "images": [],
    "annotations": [],
}
images = []
for ext in _IMG_EXTS:
    images += list(_IMG_DIR.glob(f"*{ext}"))

for idx, img in enumerate(images):

    label_path = _LABELS_DIR / img.with_suffix(".txt").name

    if not label_path.is_file():
        continue
    coco_dataset["images"].append({"file_name": img.name, "id": idx})

    img_width, img_height = Image.open(img).size
    label_data = []
    image = cv2.imread(str(img))
    for line in label_path.read_text().splitlines():
        class_id, x, y, w, h = line.split()
        class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)

        b = [x * img_width, y * img_height, w * img_width, h * img_height]
        b[0] -= 0.5 * b[2]
        b[1] -= 0.5 * b[3]
        if b[2] > 0 and b[3] > 0:
            coco_dataset["annotations"].append(
                {
                    "id": len(coco_dataset["annotations"]),
                    "bbox": b,
                    "category_id": 0,
                    "image_id": idx,
                }
            )

            cv2.rectangle(
                image,
                (int(b[0]), int(b[1])),
                (int(b[0] + b[2]), int(b[1] + b[3])),
                (0, 255, 0),
                2,
            )

    shutil.copy2(img, _IMG_SAVE_DIR / img.name)

save_path = _SAVE_DIR / "annotations.json"
save_path.write_text(json.dumps(coco_dataset, indent=2))