import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config_smoke_test import (
    BATCH_SIZE,
    CLASS_OFFSET,
    DATA_DIR,
    DEVICE,
    LEARNING_RATE,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    MOMENTUM,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    PRETRAINED,
    SEED,
    SMOKE_CHECKPOINT_DIR,
    SMOKE_LAST_MODEL_PATH,
    SMOKE_TRAIN_SAMPLES,
    SMOKE_VALID_SAMPLES,
    TRAINABLE_BACKBONE_LAYERS,
    WEIGHT_DECAY,
)
from src.dataset.dataloader import (
    build_dataloader,
    detection_collate_fn,
)
from src.faster_rcnn_model import build_model
from train import set_seed
from train import train


def random_subset(dataset, max_samples, seed):
    sample_count = min(max_samples, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    return Subset(dataset, indices)


def main():
    set_seed(SEED)

    full_train_loader, full_valid_loader = build_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        class_offset=CLASS_OFFSET,
        num_classes=NUM_CLASSES,
    )

    train_subset = random_subset(
        full_train_loader.dataset,
        SMOKE_TRAIN_SAMPLES,
        SEED,
    )
    valid_subset = random_subset(
        full_valid_loader.dataset,
        SMOKE_VALID_SAMPLES,
        SEED,
    )

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=detection_collate_fn,
        drop_last=False,
        generator=loader_generator,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    print("=" * 60)
    print("SMOKE TRAIN")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_subset)}")
    print(f"Valid samples: {len(valid_subset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
    )
    model.to(DEVICE)

    trainable_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = optim.SGD(
        trainable_params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        start_epoch=1,
        best_loss=float("inf"),
        writer=None,
        save_dir=SMOKE_CHECKPOINT_DIR,
    )

    if not SMOKE_LAST_MODEL_PATH.exists():
        raise RuntimeError(
            f"Smoke checkpoint was not created: {SMOKE_LAST_MODEL_PATH}"
        )

    print("Smoke training completed successfully.")


if __name__ == "__main__":
    main()
