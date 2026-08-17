from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YOLODetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None, class_offset=1, num_classes=4):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.class_offset = class_offset
        self.num_classes = num_classes
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        self.img_paths = sorted(path for path in self.img_dir.iterdir() if (path.is_file() and path.suffix.lower() in valid_ext))
        if not self.img_paths:
            raise RuntimeError(f"No images found in: {self.img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # 1. Load image
        img_path = self.img_paths[index]
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # 2. Load annotation
        label_path = self.label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        
        boxes = []
        labels = []
        
        # Empty .txt = valid image without objects
        if label_path.stat().st_size > 0:
            data = np.loadtxt(label_path, ndmin=2)
            if data.shape[1] != 5:
                raise ValueError(f"Invalid YOLO annotation in {label_path}. Expected 5 columns, got {data.shape[1]}")
            
            for obj in data:
                raw_cls = obj[0]
                # Validate class id
                if not np.isfinite(raw_cls):
                    raise ValueError(f"Non-finite class id in {label_path}: {raw_cls}")
                if not float(raw_cls).is_integer():
                    raise ValueError(f"Class id must be integer in {label_path}: {raw_cls}")
                cls_id = int(raw_cls) + self.class_offset
                if not (1 <= cls_id < self.num_classes):
                    raise ValueError(f"Class id out of range in {label_path}: {cls_id}. Expected foreground labels 1..{self.num_classes - 1}")
                
                # Read YOLO bbox
                xc, yc, bw, bh = obj[1:]
                coords = np.array([xc, yc, bw, bh], dtype=np.float32)
                if not np.all(np.isfinite(coords)):
                    raise ValueError(f"Non-finite bbox values in {label_path}: {coords.tolist()}")
                
                # YOLO normalized values should be [0,1]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                    raise ValueError(f"Invalid normalized bbox in {label_path}: {coords.tolist()}")
                
                # YOLO xywh -> Pascal VOC xyxy
                xmin = (xc - bw / 2) * orig_w
                ymin = (yc - bh / 2) * orig_h
                xmax = (xc + bw / 2) * orig_w
                ymax = (yc + bh / 2) * orig_h
                xmin = np.clip(xmin, 0, orig_w)
                ymin = np.clip(ymin, 0, orig_h)
                xmax = np.clip(xmax, 0, orig_w)
                ymax = np.clip(ymax, 0, orig_h)
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)

        # 3. Transforms
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # 4. Basic transformed image validation
        if not isinstance(image, torch.Tensor):
            raise TypeError("Transforms must return image as torch.Tensor")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected image shape [3,H,W], got {tuple(image.shape)}")
        if image.dtype != torch.float32:
            raise TypeError(f"Expected torch.float32 image, got {image.dtype}")

        # 5. Convert targets
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        current_h, current_w = image.shape[-2:]
        
        target = {
            "boxes": boxes, "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(boxes.shape[0], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.tensor([current_h, current_w], dtype=torch.int64),
            "path": str(img_path)
        }
        
        return image, target