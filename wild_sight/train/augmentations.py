"""The augmentations used during training and inference for the various models. These
are not meant to be an all-encompassing augmentation regime for any of the models.
Feel free to experiment with any of the available augmentations:
https://albumentations.readthedocs.io/en/latest/index.html"""

import albumentations as albu


def det_train_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.RandomResizedCrop(640, 640),
            albu.ColorJitter(0.05, 0.05, 0.05, 0.05),
            albu.RandomGamma(),
            albu.Flip(),
            albu.Transpose(),
            albu.Rotate(),
            albu.Normalize(),
        ],
        bbox_params=albu.BboxParams(
            format="pascal_voc", label_fields=["category_ids"], min_visibility=0.2
        ),
    )


def det_val_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [albu.Resize(height=height, width=width), albu.Normalize()],
        bbox_params=albu.BboxParams(
            format="pascal_voc", label_fields=["category_ids"], min_visibility=0.2
        ),
    )
