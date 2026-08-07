import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YOLODetectionDataset(Dataset):
    """
    PyTorch Dataset cho Object Detection sử dụng YOLO annotation.

    Folder structure
    ----------------
    dataset/
        train/
            images/
            labels/
        valid/
            images/
            labels/

    Label format (.txt)
    -------------------
    class x_center y_center width height
    (normalized to [0,1])

    Output
    ------
    image : Tensor [3,H,W]

    target : dict
        boxes      : FloatTensor [N,4] (xyxy)
        labels     : LongTensor [N]
        image_id   : Tensor[1]
        area       : FloatTensor[N]
        iscrowd    : LongTensor[N]
        orig_size  : Tensor[2] (H,W)
        size       : Tensor[2] (H,W)
        path       : str
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transforms=None,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

        self.img_paths = sorted([
            f for f in glob.glob(os.path.join(img_dir, "*"))
            if f.lower().endswith(valid_ext)
        ])

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in: {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        # ==================================================
        # 1. Load image
        # ==================================================
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)

        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h_img, w_img = image.shape[:2]

        # ==================================================
        # 2. Load label
        # ==================================================
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, filename + ".txt")

        boxes = []
        labels = []

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:

            data = np.loadtxt(label_path, ndmin=2)

            for obj in data:

                cls_id = int(obj[0])

                xc, yc, bw, bh = obj[1:]

                # YOLO -> Pascal VOC (xyxy)

                xmin = (xc - bw / 2) * w_img
                ymin = (yc - bh / 2) * h_img
                xmax = (xc + bw / 2) * w_img
                ymax = (yc + bh / 2) * h_img

                # Clip vào ảnh
                xmin = np.clip(xmin, 0, w_img)
                ymin = np.clip(ymin, 0, h_img)
                xmax = np.clip(xmax, 0, w_img)
                ymax = np.clip(ymax, 0, h_img)

                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)

        # ==================================================
        # 3. Albumentations
        # ==================================================
        if self.transforms is not None:

            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels,
            )

            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # ==================================================
        # 4. Convert image -> Tensor
        # ==================================================
        if not isinstance(image, torch.Tensor):

            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        # ==================================================t
        # 5. Convert boxes
        # ==================================================
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (
                (boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1])
            )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "orig_size": torch.tensor(
                [h_img, w_img],
                dtype=torch.int64
            ),
            "size": torch.tensor(
                [h_img, w_img],
                dtype=torch.int64
            ),
            "path": img_path,
        }

        return image, target