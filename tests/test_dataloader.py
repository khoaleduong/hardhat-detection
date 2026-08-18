import torch
import cv2
import numpy as np

from src.dataset.dataloader import build_test_dataloader, detection_collate_fn


def test_collate_returns_lists():
    image1 = torch.rand(3, 64, 80)
    image2 = torch.rand(3, 72, 96)

    target1 = {"boxes": torch.tensor([[1, 2, 20, 30]], dtype=torch.float32), "labels": torch.tensor([1], dtype=torch.int64)}
    target2 = {"boxes": torch.tensor([[5, 5, 40, 50]], dtype=torch.float32), "labels": torch.tensor([2], dtype=torch.int64)}

    batch = [(image1, target1), (image2, target2)]
    images, targets = detection_collate_fn(batch)

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert len(images) == 2
    assert len(targets) == 2
    assert images[0].shape == (3, 64, 80)
    assert images[1].shape == (3, 72, 96)


def test_build_test_dataloader_returns_detection_batch(tmp_path):
    image_dir = tmp_path / "test" / "images"
    label_dir = tmp_path / "test" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "sample.jpg"), image)
    (label_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )

    loader = build_test_dataloader(
        data_dir=tmp_path,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        class_offset=1,
        num_classes=4,
    )
    images, targets = next(iter(loader))

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert images[0].shape == (3, 48, 64)
    assert images[0].dtype == torch.float32
    assert targets[0]["labels"].tolist() == [1]
