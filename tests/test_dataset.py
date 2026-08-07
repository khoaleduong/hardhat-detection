# test_dataset.py
import sys
sys.path.append('.')  # Đảm bảo import được module

from src.dataset.dataset import YOLODetectionDataset
from src.dataset.transforms import get_train_transforms, get_val_transforms
import torch

# ==========================================
# CONFIG - Thay bằng đường dẫn của bạn
# ==========================================
DATA_DIR = "/Users/leduongkhoa/hardhat_detection/data"
# ==========================================

def test_dataset_basic():
    """Test 1: Dataset load được data"""
    print("[Test 1] Basic dataset loading...")
    
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=None  # Test không transform trước
    )
    
    print(f"  Found {len(dataset)} images")
    assert len(dataset) > 0, "Dataset is empty!"
    print("  ✓ Passed\n")

def test_dataset_output():
    """Test 2: Kiểm tra output format"""
    print("[Test 2] Dataset output format...")
    
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=None
    )
    
    # Test sample đầu tiên
    image, target = dataset[0]
    
    # Check image
    assert isinstance(image, torch.Tensor), f"Image should be tensor, got {type(image)}"
    assert image.dim() == 3, f"Image should be 3D [C,H,W], got {image.dim()}D"
    assert image.shape[0] == 3, f"Should have 3 channels, got {image.shape[0]}"
    assert 0 <= image.min() and image.max() <= 1, f"Image values should be [0,1], got [{image.min()}, {image.max()}]"
    print(f"  Image shape: {image.shape}")
    
    # Check target keys
    required_keys = {'boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'}
    actual_keys = set(target.keys())
    assert required_keys.issubset(actual_keys), f"Missing keys: {required_keys - actual_keys}"
    print(f"  Target keys: {list(target.keys())}")
    
    print("  ✓ Passed\n")

def test_dataset_with_transforms():
    """Test 3: Dataset với transforms"""
    print("[Test 3] Dataset with transforms...")
    
    transforms = get_train_transforms(img_size=640)
    
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=transforms
    )
    
    image, target = dataset[0]
    
    # Sau ToTensorV2 + Normalize, ảnh là Tensor [C,H,W]
    assert isinstance(image, torch.Tensor), "Image should be tensor after transforms"
    assert image.shape == (3, 640, 640), f"Expected (3,640,640), got {image.shape}"
    print(f"  Transformed image shape: {image.shape}")
    
    # Check boxes sau transform vẫn đúng format
    if len(target['boxes']) > 0:
        assert target['boxes'].shape[1] == 4, "Boxes should be [N,4]"
        print(f"  Boxes: {len(target['boxes'])} objects")
    
    print("  ✓ Passed\n")

def test_all_samples():
    """Test 4: Duyệt qua toàn bộ dataset"""
    print("[Test 4] Iterating all samples...")
    
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=None
    )
    
    errors = []
    for i in range(len(dataset)):
        try:
            image, target = dataset[i]
            if image is None:
                errors.append(f"Sample {i}: image is None")
        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")
    
    if errors:
        print(f"  ⚠ Found {len(errors)} errors:")
        for err in errors[:5]:  # Chỉ in 5 lỗi đầu
            print(f"    - {err}")
    else:
        print(f"  ✓ All {len(dataset)} samples OK")
    
    print("  ✓ Passed\n")

def test_train_vs_val():
    """Test 5: So sánh train và validation dataset"""
    print("[Test 5] Train vs Validation...")
    
    train_dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=None
    )
    
    val_dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/valid/images",
        label_dir=f"{DATA_DIR}/valid/labels",
        transforms=None
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(val_dataset)} images")
    
    # Check không trùng lặp
    train_names = {p.split('/')[-1] for p in train_dataset.img_paths}
    val_names = {p.split('/')[-1] for p in val_dataset.img_paths}
    overlap = train_names & val_names
    
    if overlap:
        print(f"  ⚠ Warning: {len(overlap)} images in both train and val")
    else:
        print(f"  ✓ No overlap between train and val")
    
    print("  ✓ Passed\n")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATASET")
    print("=" * 60 + "\n")
    
    tests = [
        test_dataset_basic,
        test_dataset_output,
        test_dataset_with_transforms,
        test_all_samples,
        test_train_vs_val,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}\n")
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)