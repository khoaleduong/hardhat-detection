import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, CLASS_NAMES, CLASS_OFFSET, NUM_CLASSES
from src.dataset.dataset import YOLODetectionDataset
from src.dataset.transforms import get_val_transforms


def visualize_sample(dataset, index):
    image, target = dataset[index]
    # CHW -> HWC
    image = image.permute(1, 2, 0).cpu().numpy()
    boxes = target["boxes"]
    labels = target["labels"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.tolist()
        class_id = label.item()
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=2)
        ax.add_patch(rect)

        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        ax.text(xmin, ymin, class_name, fontsize=10, bbox={"alpha": 0.6})

    ax.set_title(f"Index: {index} | Objects: {len(boxes)} | {target['path']}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    dataset = YOLODetectionDataset(img_dir=TRAIN_IMAGE_DIR, label_dir=TRAIN_LABEL_DIR, transforms=get_val_transforms(), class_offset=CLASS_OFFSET, num_classes=NUM_CLASSES)
    print(f"Dataset size: {len(dataset)}")

    for _ in range(10):
        index = random.randrange(len(dataset))
        visualize_sample(dataset, index)


if __name__ == "__main__":
    main()