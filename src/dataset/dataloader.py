import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import YOLODetectionDataset
from src.dataset.transforms import get_train_transforms, get_val_transforms

def yolo_collate_fn(batch):
    """Gom các ảnh thành 1 Tensor [B, C, H, W] và giữ targets dưới dạng list(dict)."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    assert images.dim() == 4, f"Expected 4D tensor, got {images.dim()}D"
    return images, list(targets)

def build_dataloader(data_dir, img_size=640, batch_size=64, num_workers=4, pin_memory=True):
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset = YOLODetectionDataset(
        img_dir=f"{data_dir}/train/images",
        label_dir=f"{data_dir}/train/labels",
        transforms=train_transforms
    )

    val_dataset = YOLODetectionDataset(
        img_dir=f"{data_dir}/train/images",
        label_dir=f"{data_dir}/train/labels",
        transforms=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolo_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolo_collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader