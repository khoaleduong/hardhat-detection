import argparse
from pathlib import Path

import cv2
import torch

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    TRAINABLE_BACKBONE_LAYERS,
)
from src.checkpoint import load_checkpoint
from src.faster_rcnn_model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Faster R-CNN inference on one image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prediction.jpg"),
        help="Output image path (default: prediction.jpg)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    original_image = cv2.imread(str(args.image))
    if original_image is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_tensor = (
        torch.from_numpy(rgb_image)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        / 255.0
    )

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
    )
    model.to(DEVICE)
    load_checkpoint(
        checkpoint_path=BEST_MODEL_PATH,
        model=model,
        device=DEVICE,
    )
    model.eval()

    with torch.no_grad():
        output = model([image_tensor.to(DEVICE)])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    detections = []
    for box, label_tensor, score_tensor in zip(boxes, labels, scores):
        label = int(label_tensor.item())
        score = float(score_tensor.item())
        if score < args.score_threshold or not 1 <= label < len(CLASS_NAMES):
            continue

        box_values = box.tolist()
        detections.append((CLASS_NAMES[label], score, box_values))

        x1, y1, x2, y2 = (round(value) for value in box_values)
        text = f"{CLASS_NAMES[label]} {score:.2f}"
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            original_image,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), original_image):
        raise RuntimeError(f"Cannot write output image: {args.output}")

    print(f"Image: {args.image}")
    print(f"Device: {DEVICE}")
    print(f"Threshold: {args.score_threshold}")
    print(f"Detections kept: {len(detections)}")
    for class_name, score, box in detections:
        formatted_box = [round(value, 1) for value in box]
        print(f"{class_name:<8} score={score:.2f} box={formatted_box}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
