import cv2
import numpy as np
import pytest
import torch
from src.dataset.dataset import YOLODetectionDataset
from src.dataset.transforms import get_val_transforms


def create_sample(tmp_path, label_text, image_shape=(100, 200, 3)):
    img_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    img_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros(image_shape, dtype=np.uint8)
    img_path = img_dir / "sample.jpg"
    cv2.imwrite(str(img_path), image)

    label_path = label_dir / "sample.txt"
    label_path.write_text(label_text, encoding="utf-8")

    return img_dir, label_dir


def test_normal_sample(tmp_path):
    img_dir, label_dir = create_sample(tmp_path, "0 0.5 0.5 0.4 0.4\n")
    dataset = YOLODetectionDataset(img_dir=img_dir, label_dir=label_dir, transforms=get_val_transforms(), class_offset=1, num_classes=4)
    image, target = dataset[0]

    assert image.dtype == torch.float32
    assert image.ndim == 3
    assert image.shape[0] == 3
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)
    # YOLO class 0 -> Faster RCNN 1
    assert target["labels"][0].item() == 1


def test_yolo_box_conversion(tmp_path):
    img_dir, label_dir = create_sample(tmp_path, "0 0.5 0.5 0.4 0.4\n")
    dataset = YOLODetectionDataset(img_dir=img_dir, label_dir=label_dir, transforms=get_val_transforms(), class_offset=1, num_classes=4)
    _, target = dataset[0]

    expected = torch.tensor([[60.0, 30.0, 140.0, 70.0]], dtype=torch.float32)
    assert torch.allclose(target["boxes"], expected)


def test_empty_annotation(tmp_path):
    img_dir, label_dir = create_sample(tmp_path, "")
    dataset = YOLODetectionDataset(img_dir=img_dir, label_dir=label_dir, transforms=get_val_transforms(), class_offset=1, num_classes=4)
    _, target = dataset[0]

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["area"].shape == (0,)


def test_invalid_non_integer_class(tmp_path):
    img_dir, label_dir = create_sample(tmp_path, "1.7 0.5 0.5 0.4 0.4\n")
    dataset = YOLODetectionDataset(img_dir=img_dir, label_dir=label_dir, transforms=get_val_transforms(), class_offset=1, num_classes=4)

    with pytest.raises(ValueError):
        dataset[0]


def test_out_of_range_class(tmp_path):
    img_dir, label_dir = create_sample(tmp_path, "10 0.5 0.5 0.4 0.4\n")
    dataset = YOLODetectionDataset(img_dir=img_dir, label_dir=label_dir, transforms=get_val_transforms(), class_offset=1, num_classes=4)

    with pytest.raises(ValueError):
        dataset[0]