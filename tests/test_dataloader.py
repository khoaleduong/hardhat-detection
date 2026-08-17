import torch
from src.dataset.dataloader import detection_collate_fn


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