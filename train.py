import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, CLASS_OFFSET, NUM_CLASSES, PRETRAINED, TRAINABLE_BACKBONE_LAYERS, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, LR_FACTOR, LR_PATIENCE, LR_MIN, DEVICE, CHECKPOINT_DIR, LAST_MODEL_PATH, SEED
from src.dataset.dataloader import build_dataloader
from src.faster_rcnn_model import build_model
from src.checkpoint import load_checkpoint, save_checkpoint
from src.engine import train_one_epoch, validate_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, valid_loader, optimizer, scheduler, device, num_epochs, start_epoch=1, best_loss=float("inf"), writer=None, save_dir=CHECKPOINT_DIR):
    import shutil
    import time

    save_dir.mkdir(parents=True, exist_ok=True)
    history = {"train": [], "valid": []}

    print("=" * 70)
    print(f"Start Training from Epoch {start_epoch} to {num_epochs}")
    print("=" * 70)

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()

        # 1. Train
        train_metrics = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device, epoch=epoch, writer=writer)

        # 2. Validation loss
        valid_metrics = validate_loss(model=model, loader=valid_loader, device=device, epoch=epoch, writer=writer)

        history["train"].append(train_metrics)
        history["valid"].append(valid_metrics)

        # 3. Scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_metrics["loss"])
            else:
                scheduler.step()

        # 4. Checkpoint
        val_loss = valid_metrics["loss"]
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        last_path = save_dir / "last_model.pth"
        best_path = save_dir / "best_model.pth"

        save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_loss=best_loss, save_path=last_path)

        if is_best:
            shutil.copyfile(last_path, best_path)
            print(f"Best model updated (val_loss={best_loss:.4f})")

        # 5. Epoch summary
        elapsed = time.time() - start_time
        print(f"[{epoch:03d}/{num_epochs:03d}] Train Loss: {train_metrics['loss']:.4f} | Valid Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
        print("-" * 70)

    print()
    print("Training Finished.")
    print(f"Best Validation Loss: {best_loss:.4f}")

    if writer is not None:
        writer.close()

    return history


def main():
    # 1. Reproducibility
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data
    train_loader, valid_loader = build_dataloader(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, class_offset=CLASS_OFFSET, num_classes=NUM_CLASSES)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")

    # 3. Model
    model = build_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED, trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS)
    model.to(DEVICE)

    # 4. Optimizer
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # 5. Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

    # 6. TensorBoard
    writer = SummaryWriter(log_dir="runs/hardhat_detection")

    # 7. Resume checkpoint
    start_epoch = 1
    best_loss = float("inf")

    if LAST_MODEL_PATH.exists():
        print(f"Checkpoint found: {LAST_MODEL_PATH}")
        last_epoch, best_loss = load_checkpoint(checkpoint_path=LAST_MODEL_PATH, model=model, optimizer=optimizer, scheduler=scheduler, device=DEVICE)
        start_epoch = last_epoch + 1
        print(f"Resume training from epoch {start_epoch}")

    # 8. Train
    train(model=model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, scheduler=scheduler, device=DEVICE, num_epochs=NUM_EPOCHS, start_epoch=start_epoch, best_loss=best_loss, writer=writer, save_dir=CHECKPOINT_DIR)


if __name__ == "__main__":
    main()