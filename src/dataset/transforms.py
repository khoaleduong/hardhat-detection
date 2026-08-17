import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # uint8 [0,255] -> float32 [0,1]
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_val_transforms():
    return A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )