import torch
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, CLASS_OFFSET, NUM_CLASSES, CLASS_NAMES, TRAINABLE_BACKBONE_LAYERS, BEST_MODEL_PATH, DEVICE
from src.faster_rcnn_model import build_model
from src.dataset.dataloader import build_dataloader
from src.checkpoint import load_checkpoint
from src.engine import evaluate


def main():
    print("=" * 60)
    print("Evaluating best checkpoint")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {BEST_MODEL_PATH}")
    print("=" * 60)

    # 1. Validation DataLoader
    _, valid_loader = build_dataloader(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, class_offset=CLASS_OFFSET, num_classes=NUM_CLASSES)
    print(f"Validation samples: {len(valid_loader.dataset)}") # type: ignore

    # 2. Build model
    model = build_model(num_classes=NUM_CLASSES, pretrained=False, trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS)
    model.to(DEVICE)

    # 3. Load best checkpoint
    load_checkpoint(checkpoint_path=BEST_MODEL_PATH, model=model, device=DEVICE) # type: ignore

    # 4. Evaluate
    evaluate(model=model, loader=valid_loader, device=DEVICE, class_names=CLASS_NAMES)


if __name__ == "__main__":
    main()