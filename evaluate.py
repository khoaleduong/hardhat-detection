import torch
from src.faster_rcnn_model import build_model
from src.dataset.dataloader import build_dataloader
from src.checkpoint import load_checkpoint
from src.engine import evaluate
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, CLASS_OFFSET, NUM_CLASSES, CLASS_NAMES, PRETRAINED, TRAINABLE_BACKBONE_LAYERS, BEST_MODEL_PATH, DEVICE


CLASS_NAMES = ["background", "helmet", "head", "person"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    _, valid_loader = build_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        class_offset=CLASS_OFFSET,
        num_classes=NUM_CLASSES,
    )

    # Model
    model = build_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED, trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS)
    model.to(device)

    # Load best checkpoint
    load_checkpoint(checkpoint_path="checkpoints/best_model.pth", model=model, device=device)  # type: ignore

    # Evaluation
    evaluate(model=model, loader=valid_loader, device=device, class_names=CLASS_NAMES)


if __name__ == "__main__":
    main()