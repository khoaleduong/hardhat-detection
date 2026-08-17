from torch.utils.data import DataLoader
from src.dataset.dataset import YOLODetectionDataset
from src.dataset.transforms import get_train_transforms, get_val_transforms


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloader(data_dir, batch_size, num_workers, pin_memory, class_offset, num_classes):
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset = YOLODetectionDataset(
        img_dir=data_dir / "train" / "images",
        label_dir=data_dir / "train" / "labels",
        transforms=train_transforms,
        class_offset=class_offset,
        num_classes=num_classes,
    )

    val_dataset = YOLODetectionDataset(
        img_dir=data_dir / "valid" / "images",
        label_dir=data_dir / "valid" / "labels",
        transforms=val_transforms,
        class_offset=class_offset,
        num_classes=num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader