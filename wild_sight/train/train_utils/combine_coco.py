#!/usr/bin/env python3
"""A script to combine COCO datasets into one.

./wild_sight/train/train_utils/combine_coco.py \
    --coco_metadata_path "~/Downloads/whaleshark.coco/annotations/instances_train2020.json,/media/alex/Elements/gzgc.coco/annotations/instances_train2020.json,/home/alex/datasets/coco/instances_train2017.json,/home/alex/datasets/coco/instances_val2017.json" \
    --coco_image_dirs "~/Downloads/whaleshark.coco/images/train2020,/media/alex/Elements/gzgc.coco/images/train2020,/home/alex/datasets/coco/images/train2017,/home/alex/datasets/coco/images/val2017" \
    --save_dir ~/datasets/whale-giraffe-zebra-coco


Experiment 1:
Train a model on everything but COCO.

./wild_sight/train/train_utils/combine_coco.py \
    --coco_metadata_path "~/Downloads/whaleshark.coco/annotations/instances_train2020.json,/media/alex/Elements/gzgc.coco/annotations/instances_train2020.json" \
    --coco_image_dirs "~/Downloads/whaleshark.coco/images/train2020,/media/alex/Elements/gzgc.coco/images/train2020,/home/alex/Downloads/archive (2)/shark-images" \
    --save_dir ~/datasets/whale-giraffe-zebra

./wild_sight/train/train_utils/combine_coco.py \
    --coco_metadata_path "/home/alex/datasets/swift-parrots/dataset/annotations.json" \
    --coco_image_dirs "/home/alex/datasets/swift-parrots/dataset/images,/home/alex/datasets/birds" \
    --save_dir ~/datasets/swift-parrot-and-birds

./wild_sight/train/train_utils/combine_coco.py \
    --coco_metadata_path "/home/alex/datasets/swift-parrots/dataset-withoutails/annotations.json" \
    --coco_image_dirs "/home/alex/datasets/swift-parrots/dataset-withoutails/images,/home/alex/datasets/birds" \
    --save_dir ~/datasets/swift-parrot-and-birds-no-tails
"""

import argparse
import pathlib
import json
import shutil
import hashlib
from typing import List


CLASSES = {"parrot": False}
CATEGORIES = []

def merge_datasets(
    metadata_paths: List[pathlib.Path],
    image_dirs: List[pathlib.Path],
    save_dir: pathlib.Path,
):
    images = []
    annotations = []
    categories = []

    image_dirs_label = []
    for label_path, image_dir in zip(metadata_paths, image_dirs):
        new_image_dir = save_dir / "images" / image_dir.parent.parent.name
        new_image_dir.mkdir(exist_ok=True, parents=True)
        print(metadata_paths)
        metadata = json.loads(label_path.expanduser().read_text())
        image_id_offset = len(images)
        category_id_offset = len(categories)
        image_dirs_label.append(image_dir)
        internal_category_map = {}
        for category in metadata.get("categories", []):
            for data_class in CLASSES:
                if data_class in category["name"]:
                    if not CLASSES[data_class]:
                        new_id = category["id"] + category_id_offset
                        internal_category_map[category["id"]] = new_id
                        category["id"] = new_id
                        categories.append(category)
                        CATEGORIES.append(data_class)
                        CLASSES[data_class] = True
                    else:
                        internal_category_map[category["id"]] = CATEGORIES.index(data_class)


        print(CATEGORIES)

        print(categories)
        internal_image_map = {}
        for image in metadata["images"]:
            new_id = image["id"] + image_id_offset
            internal_image_map[image["id"]] = new_id
            image["id"] = new_id
            new_file = new_image_dir / image["file_name"]
            if not new_file.is_file():
                shutil.copy2(
                    image_dir / image["file_name"], new_image_dir / image["file_name"]
                )
            image["file_name"] = f"{new_image_dir.name}/{image['file_name']}"
            images.append(image)

        # Annotations must update image id reference and category
        for annotation in metadata["annotations"]:
            annotation["image_id"] = internal_image_map[annotation["image_id"]]
            if annotation["category_id"] in internal_category_map:
                annotation["category_id"] = internal_category_map[annotation["category_id"]]
                if "segmentation" in annotation:
                    del annotation["segmentation"]
                annotations.append(annotation)
                print(label_path, annotation)

    # Get all the image dirs that are specified without corresponding labels.
    # Simply add these to the dataset's image list
    no_label_image_dirs = [
        image_dir for image_dir in image_dirs if image_dir not in image_dirs_label
    ]
    for image_dir in no_label_image_dirs:
        new_image_dir = save_dir / "images" / image_dir.name
        new_image_dir.mkdir(exist_ok=True, parents=True)
        for ext in ["jpg", "jpeg", "JPG", "jpeg"]:
            for image in image_dir.rglob(f"*.{ext}"):
                image_dict = {
                    "id": len(images),
                    "file_name": str(image)
                }
                images.append(image_dict)
                new_file = new_image_dir / f"{hashlib.sha256(bytes(image)).hexdigest()}{image.suffix}"
                print(image, new_file)
                if not new_file.is_file():
                    shutil.copy2(
                        image, new_file
                    )
    
    final_cats = set([anno["category_id"] for anno in annotations if "category_id" in anno])
    assert len(final_cats) == len(CLASSES)

    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "annotations.json"
    save_path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--coco_metadata_paths",
        required=True,
        type=str,
        help="Comma separated list of metadata paths.",
    )
    parser.add_argument(
        "--coco_image_dirs",
        required=True,
        type=str,
        help="Comma separated list of metadata paths.",
    )
    parser.add_argument("--save_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    metadata_paths = [
        pathlib.Path(path).expanduser() for path in args.coco_metadata_paths.split(",")
    ]
    image_dirs = [pathlib.Path(path).expanduser() for path in args.coco_image_dirs.split(",")]
    merge_datasets(metadata_paths, image_dirs, args.save_dir.expanduser())
