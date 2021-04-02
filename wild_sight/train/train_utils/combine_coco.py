#!/usr/bin/env python3
"""A script to combine COCO datasets into one."""

import argparse
import pathlib
import json
import shutil
from typing import List


def merge_datasets(
    metadata_paths: List[pathlib.Path],
    image_dirs: List[pathlib.Path],
    save_dir: pathlib.Path,
):
    images = []
    annotations = []
    categories = []
    for label_path, image_dir in zip(metadata_paths, image_dirs):
        new_image_dir = save_dir / "images" / image_dir.parent.parent.name
        new_image_dir.mkdir(exist_ok=True, parents=True)
        metadata = json.loads(label_path.read_text())
        image_id_offset = len(images)
        category_id_offset = len(categories)

        internal_category_map = {}
        for category in metadata["categories"]:
            new_id = category["id"] + category_id_offset
            internal_category_map[category["id"]] = new_id
            category["id"] = new_id
            categories.append(category)

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
            annotation["category_id"] = internal_category_map[annotation["category_id"]]
            annotations.append(annotation)

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
        pathlib.Path(path) for path in args.coco_metadata_paths.split(",")
    ]
    image_dirs = [pathlib.Path(path) for path in args.coco_image_dirs.split(",")]
    merge_datasets(metadata_paths, image_dirs, args.save_dir.expanduser())