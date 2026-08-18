import os
from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = Path(
    os.getenv(
        "CHECKPOINT_DIR",
        ROOT_DIR / "checkpoints",
    )
)

TRAIN_IMAGE_DIR = DATA_DIR / "train" / "images"
TRAIN_LABEL_DIR = DATA_DIR / "train" / "labels"

VALID_IMAGE_DIR = DATA_DIR / "valid" / "images"
VALID_LABEL_DIR = DATA_DIR / "valid" / "labels"

TEST_IMAGE_DIR = DATA_DIR / "test" / "images"
TEST_LABEL_DIR = DATA_DIR / "test" / "labels"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET
# ============================================================

CLASS_NAMES = [
    "background",
    "head",
    "helmet",
    "person",
]

NUM_CLASSES = len(CLASS_NAMES)

# YOLO 0,1,2 -> Faster R-CNN 1,2,3
CLASS_OFFSET = 1


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 12
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

SHUFFLE_TRAIN = True
SHUFFLE_VALID = False


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "fasterrcnn_resnet50_fpn"

PRETRAINED = True

TRAINABLE_BACKBONE_LAYERS = 2


# ============================================================
# TRAINING
# ============================================================

NUM_EPOCHS = 10

LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


# ============================================================
# SCHEDULER
# ============================================================

SCHEDULER_NAME = "reduce_on_plateau"

LR_FACTOR = 0.1
LR_PATIENCE = 2
LR_MIN = 1e-6


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECKPOINT
# ============================================================

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42
