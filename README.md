# Hardhat Detection

Hardhat Detection is an object detection project for identifying heads, helmets, and people in workplace images. It uses TorchVision's Faster R-CNN with a ResNet-50 FPN backbone and a dataset stored in YOLO annotation format.

The repository contains dataset preparation, augmentation, training, checkpoint resume, test-set evaluation, and single-image inference code. Training supports CUDA automatic mixed precision (AMP) and falls back to standard FP32 execution on CPU.

## Model

The model is built with `torchvision.models.detection.fasterrcnn_resnet50_fpn`. During training, it starts from TorchVision's default pretrained Faster R-CNN weights and replaces the ROI box predictor with a four-class predictor. Two backbone layers are trainable by default.

The class mapping is:

| ID | Faster R-CNN class | YOLO source ID |
|---:|---|---:|
| 0 | background | — |
| 1 | head | 0 |
| 2 | helmet | 1 |
| 3 | person | 2 |

`CLASS_OFFSET = 1` converts YOLO foreground IDs to the labels expected by Faster R-CNN, where class `0` is reserved for background.

## Project structure

```text
hardhat-detection/
├── config.py                  # Paths, classes, and training configuration
├── config_smoke_test.py       # Overrides used by the smoke training run
├── train.py                   # Full training and automatic resume
├── train_smoke.py             # Small deterministic training run
├── evaluate.py                # Test-set evaluation
├── predict.py                 # Single-image inference and visualization
├── pyproject.toml             # Project dependencies and uv configuration
├── scripts/
│   ├── download_data.py
│   ├── visualize_dataset.py
│   └── visualize_train_augmentation.py
├── src/
│   ├── checkpoint.py
│   ├── engine.py
│   ├── faster_rcnn_model.py
│   └── dataset/
│       ├── dataloader.py
│       ├── dataset.py
│       └── transforms.py
├── tests/
└── notebooks/
```

## Dataset

The expected dataset layout is:

```text
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each image must have a label file with the same stem. Labels use normalized YOLO coordinates:

```text
class_id x_center y_center width height
```

The dataset converts these boxes to absolute `xyxy` coordinates and returns the TorchVision detection contract:

```text
image:   float32 Tensor[C, H, W] in [0, 1]
target:  boxes[N, 4], labels[N], area[N], iscrowd[N], and image metadata
```

Empty label files are accepted as images without objects. Missing labels, invalid class IDs, and malformed boxes raise an error. Detection batches remain `list[Tensor]` and `list[dict]` so images can have different spatial sizes.

Training augmentation consists of horizontal flipping and random brightness/contrast adjustment. Validation and test transforms only convert pixels to `float32` tensors. Images are not resized or ImageNet-normalized by Albumentations; Faster R-CNN performs its own resizing and normalization internally.

### Downloading the dataset

The included Roboflow downloader reads `ROBOFLOW_API_KEY` from the environment. A local `.env` file can be used:

```dotenv
ROBOFLOW_API_KEY=your_api_key
```

Then run:

```bash
python scripts/download_data.py
```

The script downloads version 1 of the configured Roboflow project in YOLOv8 format into `data/`. Both `data/` and `.env` are excluded from Git; do not commit the API key.

The dataset and training augmentations can be inspected with:

```bash
python scripts/visualize_dataset.py
python scripts/visualize_train_augmentation.py
```

## Installation

Python 3.10 or newer is required. The repository uses `uv` and pins local PyTorch packages to the CUDA 13.0 PyTorch index through the `local-gpu` dependency group.

For local GPU development and tests:

```bash
uv sync --group local-gpu --group test
```

Commands can then be run through `uv`, for example:

```bash
uv run pytest
uv run python train_smoke.py
```

The local PyTorch index is environment-specific. Hosted GPU environments such as Google Colab should normally keep their preinstalled compatible Torch and TorchVision builds; see the Colab section below.

## Configuration

The main settings are defined in `config.py`:

| Setting | Current value or behavior |
|---|---|
| Dataset directory | `<project_root>/data` |
| Checkpoint directory | `$CHECKPOINT_DIR`, otherwise `<project_root>/checkpoints` |
| Classes | background, head, helmet, person |
| Batch size | 16 |
| DataLoader workers | 2 |
| Epochs | 10 |
| Optimizer | SGD, learning rate `0.005`, momentum `0.9`, weight decay `0.0005` |
| Scheduler | ReduceLROnPlateau, factor `0.1`, patience `2`, minimum LR `1e-6` |
| Device | CUDA when available, otherwise CPU |

Pinned memory is enabled only when CUDA is available. DataLoader workers remain persistent when `num_workers > 0`.

`DATA_DIR` does not currently have an environment-variable override. `CHECKPOINT_DIR` does. Set it before importing `config` or starting an entry point:

```bash
export CHECKPOINT_DIR=/path/to/checkpoints
python train.py
```

PowerShell equivalent:

```powershell
$env:CHECKPOINT_DIR = "D:\path\to\checkpoints"
python train.py
```

## Training

Run full training with:

```bash
python train.py
```

Training uses SGD and reduces the learning rate based on validation loss. CUDA runs use `torch.autocast` and `torch.amp.GradScaler`; CPU runs use FP32. Images and tensor targets use non-blocking device transfers, which work with pinned DataLoader memory on CUDA.

Validation loss is computed with the detector in training mode because TorchVision Faster R-CNN only returns its loss dictionary in that mode. Gradients remain disabled during validation.

TensorBoard logs are written to `runs/hardhat_detection`:

```bash
tensorboard --logdir runs/hardhat_detection
```

If `last_model.pth` exists, training restores the model, optimizer, scheduler, epoch, best validation loss, and GradScaler state when available, then resumes at the next epoch. Older checkpoints without a scaler state remain loadable.

## Smoke training

Before a full run, execute:

```bash
python train_smoke.py
```

The smoke configuration uses batch size 1, zero workers, two epochs, and deterministic random subsets of at most 64 training and 16 validation samples. It starts from epoch 1 every time and writes only to `checkpoints/smoke/`, so it does not resume from or overwrite the main training checkpoints.

## Evaluation

Evaluate the best checkpoint on the test split:

```bash
python evaluate.py
```

The script builds the model without downloading pretrained weights, loads `best_model.pth`, and evaluates every sample in `data/test`. It reports:

- mAP@50:95
- mAP@50
- mAP@75
- mAR@1
- mAR@10
- mAR@100
- AP per class

Metrics are computed with `torchmetrics.detection.MeanAveragePrecision` using `xyxy` boxes. All three foreground classes, including `person`, are included.

## Prediction

Run inference on one image with:

```bash
python predict.py path/to/image.jpg --score-threshold 0.5 --output prediction.jpg
```

The command loads `best_model.pth`, filters detections by score, draws valid foreground boxes, and saves the annotated image. The default threshold is `0.5`, and the default output path is `prediction.jpg`. The input is converted from OpenCV BGR to RGB and then to a CHW `float32` tensor in `[0, 1]`; resizing and normalization remain inside Faster R-CNN.

## Checkpoints

By default, full training writes:

```text
checkpoints/
├── last_model.pth
└── best_model.pth
```

- `last_model.pth` is replaced after every epoch and is used for automatic resume.
- `best_model.pth` is updated when validation loss reaches a new minimum.

Each checkpoint stores the model and optimizer states, epoch, and best validation loss. Scheduler and GradScaler states are included when present. Checkpoint writes use a temporary file followed by replacement. The `checkpoints/` directory is excluded from Git.

## Google Colab

Colab provides its own matched PyTorch, TorchVision, and CUDA environment. Do not install the repository's `local-gpu` dependency group solely to match the CUDA build used on a local machine.

A minimal setup matching the included Colab notebook is:

```python
!git clone https://github.com/khoaleduong/hardhat-detection.git
%cd hardhat-detection

!pip install -q uv
!uv pip install --system -r pyproject.toml
```

Set `ROBOFLOW_API_KEY` in the notebook environment, then download the dataset without placing the key in source code:

```python
import os

os.environ["ROBOFLOW_API_KEY"] = "<read this value from a Colab secret>"
!python scripts/download_data.py
```

To keep checkpoints on Google Drive, mount Drive and set `CHECKPOINT_DIR` before importing project configuration or running training:

```python
from google.colab import drive
import os

drive.mount("/content/drive")
os.environ["CHECKPOINT_DIR"] = (
    "/content/drive/MyDrive/hardhat-detection/checkpoints"
)
```

Then start training normally:

```python
!python train.py
```

If `last_model.pth` already exists in that Drive directory, training resumes automatically.

## Reproducibility

Training seeds Python, NumPy, PyTorch, and all CUDA devices with `SEED = 42`. Smoke subsets also use this seed. The project does not enable deterministic CUDA algorithms, so identical results across machines or GPU configurations are not guaranteed.
