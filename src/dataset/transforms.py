import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=640):
    """Transforms for training"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams( # type: ignore
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))

def get_val_transforms(img_size=640):
    """Transforms for validation"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams( # type: ignore
        format='pascal_voc',
        label_fields=['class_labels']
    ))