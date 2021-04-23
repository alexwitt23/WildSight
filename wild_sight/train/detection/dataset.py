""" Datasets for loading data for our various training regimes. """

from typing import Tuple
import pathlib
import json
import random

import cv2
import torch

from wild_sight.train import augmentations as augs


class DetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_width: int,
        img_height: int,
        img_ext: str = ".png",
        validation: bool = False,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())

        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.img_height = img_height
        self.img_width = img_width
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )
        self.images = {}
        for image in self.meta_data["images"]:
            self.images[image["id"]] = {
                "file_name": data_dir / image["file_name"],
                "annotations": [],
            }

        if validation:
            num_val = int(len(self.images) * 0.2)
            image_ids = list(self.images.keys())
            random.shuffle(image_ids)
            keep_ids = image_ids[:num_val]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }

        for anno in self.meta_data["annotations"]:
            if anno["image_id"] in self.images:
                self.images[anno["image_id"]]["annotations"].append(anno)
        self.ids_map = {idx: img_id for idx, img_id in enumerate(self.images.keys())}
        self.len = len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = None
        while image is None:
            image_data = self.images[self.ids_map[idx]]
            image = cv2.imread(str(image_data["file_name"]))
            idx = random.choice(range(self.__len__()))

        boxes = []
        category_ids = []
        for anno in image_data["annotations"]:
            box = torch.Tensor(anno["bbox"])
            box[2:] += box[:2]
            box[0::2].clamp_(0, image.shape[1])
            box[1::2].clamp_(0, image.shape[0])
            boxes.append(box)
            category_ids.append(anno["category_id"])

        return self.transform(
            image=image,
            bboxes=boxes,
            category_ids=category_ids,
            image_ids=self.ids_map[idx],
        )

    def __len__(self) -> int:
        return self.len


class AfricanWildlife(torch.utils.data.Dataset):
    """
    https://www.kaggle.com/biancaferreira/african-wildlife
    """

    def __init__(
        self,
        data_dir: pathlib.Path,
        img_width: int,
        img_height: int,
        validation: bool = False,
        **kwargs,
    ) -> None:
        assert data_dir.is_dir()

        self.images = list(data_dir.rglob("*.jpg"))
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )

        self.classes = {}
        for img in self.images:
            labels = img.with_suffix(".txt").read_text()
            for label in labels.splitlines():
                class_id, *_ = label.split(" ")
                class_id = int(class_id)

                if class_id in self.classes:
                    self.classes[class_id].append(img)
                else:
                    self.classes[class_id] = [img]

        max_imgs_class = max(len(imgs) for imgs in self.classes.values())
        self.virtual_num_imgs = len(self.classes) * max_imgs_class

    def __getitem__(self, idx: int):

        class_idx = idx % len(self.classes)

        image_path = random.choice(self.classes[class_idx])
        image = cv2.imread(str(image_path))
        assert image is not None, image_path
        height, width = image.shape[:2]

        labels = image_path.with_suffix(".txt").read_text()
        category_ids = []
        boxes = []
        for label in labels.splitlines():
            class_id, x_c, y_c, w, h = label.split(" ")
            class_id, x_c, y_c, w, h = (
                int(class_id),
                float(x_c),
                float(y_c),
                float(w),
                float(h),
            )
            category_ids.append(class_id)

            x0 = x_c - (0.5 * w)
            y0 = y_c - (0.5 * h)
            x1 = x0 + w
            y1 = y0 + h
            box = torch.Tensor([x0, y0, x1, y1])
            box[0::2].clamp_(0.0, 1.0)
            box[1::2].clamp_(0.0, 1.0)
            box *= torch.Tensor([width, height, width, height])
            boxes.append(box)

        return self.transform(
            image=image, bboxes=boxes, category_ids=category_ids, image_ids=idx,
        )

    def __len__(self) -> int:
        return len(self.images)


class GZGC(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_width: int,
        img_height: int,
        validation: bool = False,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())

        self.images = list(data_dir.glob(f"*.jpg"))
        assert self.images, f"No images found in {data_dir}."

        self.img_height = img_height
        self.img_width = img_width
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )
        self.images = {}
        for image in self.meta_data["images"]:
            self.images[image["id"]] = {
                "file_name": data_dir / image["file_name"],
                "annotations": [],
            }

        image_ids = list(self.images.keys())
        random.Random(42).shuffle(image_ids)
        if validation:
            num_val = int(len(self.images) * 0.2)
            keep_ids = image_ids[-num_val:]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }
        else:
            num_val = int(len(self.images) * 0.2)
            keep_ids = image_ids[:-num_val]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }

        for anno in self.meta_data["annotations"]:
            if anno["image_id"] in self.images:
                self.images[anno["image_id"]]["annotations"].append(anno)
        self.ids_map = {idx: img_id for idx, img_id in enumerate(self.images.keys())}
        self.len = len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = None
        while image is None:
            image_data = self.images[self.ids_map[idx]]
            image = cv2.imread(str(image_data["file_name"]))
            idx = random.choice(range(self.__len__()))

        boxes = []
        category_ids = []
        for anno in image_data["annotations"]:
            box = torch.Tensor(anno["bbox"])
            box[2:] += box[:2]
            box[0::2].clamp_(0, image.shape[1])
            box[1::2].clamp_(0, image.shape[0])
            boxes.append(box)
            category_ids.append(anno["category_id"])

        return self.transform(
            image=image,
            bboxes=boxes,
            category_ids=category_ids,
            image_ids=self.ids_map[idx],
        )

    def __len__(self) -> int:
        return self.len

    def __str__(self) -> str:
        return f"{len(self.images)} images."


class WhaleShark(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_width: int,
        img_height: int,
        validation: bool = False,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())

        self.images = list(data_dir.rglob("*.jpg"))
        assert self.images, f"No images found in {data_dir}."

        self.img_height = img_height
        self.img_width = img_width
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )
        self.images = {}
        for image in self.meta_data["images"]:
            self.images[image["id"]] = {
                "file_name": data_dir / image["file_name"],
                "annotations": [],
            }

        image_ids = list(self.images.keys())
        random.Random(42).shuffle(image_ids)
        if validation:
            num_val = int(len(self.images) * 0.5)
            keep_ids = image_ids[-num_val:]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }
        else:
            num_val = int(len(self.images) * 0.5)
            keep_ids = image_ids[:-num_val]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }

        for anno in self.meta_data["annotations"]:
            if anno["image_id"] in self.images:
                self.images[anno["image_id"]]["annotations"].append(anno)
        self.ids_map = {idx: img_id for idx, img_id in enumerate(self.images.keys())}
        self.len = len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_data = self.images[self.ids_map[idx]]
        image = cv2.imread(str(image_data["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = []
        category_ids = []
        for anno in image_data["annotations"]:
            if "bbox" in anno:
                box = torch.Tensor(anno["bbox"])
                box[2:] += box[:2]
                box[0::2].clamp_(0, image.shape[1])
                box[1::2].clamp_(0, image.shape[0])
                if (box[2] - box[0]) * (box[3] - box[1]):
                    boxes.append(box)
                    category_ids.append(anno["category_id"])

        return self.transform(
            image=image,
            bboxes=boxes,
            category_ids=category_ids,
            image_ids=self.ids_map[idx],
        )

    def __len__(self) -> int:
        return self.len

    def __str__(self) -> str:
        return f"{len(self.images)} images."
