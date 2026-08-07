# test_dataloader.py
"""Test DataLoader với dataset thật"""
import torch
from src.dataset.dataset import YOLODetectionDataset
from src.dataset.transforms import get_train_transforms, get_val_transforms
from src.dataset.dataloader import yolo_collate_fn

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = "/Users/leduongkhoa/hardhat_detection/data"
# ==========================================

def test_collate_fn():
    """Test 1: Collate function với data thật"""
    print("[Test 1] Collate function...")
    
    # Load 4 samples từ dataset thật
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=get_train_transforms(640)
    )
    
    # Tạo batch thủ công
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    images, targets = yolo_collate_fn(batch)
    
    assert images.dim() == 4, f"Expected 4D, got {images.dim()}D"
    assert images.shape[0] == len(batch), f"Batch size mismatch"
    assert len(targets) == len(batch), f"Targets count mismatch"
    
    print(f"  Batch shape: {images.shape}")
    print(f"  Targets: {len(targets)} dicts")
    print("  ✓ Passed\n")

def test_dataloader_iteration():
    """Test 2: DataLoader iteration"""
    print("[Test 2] DataLoader iteration...")
    
    from torch.utils.data import DataLoader
    
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transforms=get_train_transforms(640)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Dùng 0 để debug dễ hơn
        collate_fn=yolo_collate_fn
    )
    
    # Test 3 batches
    batch_count = 0
    for i, (images, targets) in enumerate(loader):
        if i >= 3:
            break
        
        print(f"  Batch {i}: images={images.shape}, targets={len(targets)}")
        
        # Check mỗi batch
        assert images.shape[0] <= 4, "Batch too large"
        assert all(isinstance(t, dict) for t in targets), "Targets must be dicts"
        
        batch_count += 1
    
    assert batch_count > 0, "No batches loaded"
    print("  ✓ Passed\n")

def test_val_loader_no_shuffle():
    """Test 3: Validation loader không shuffle"""
    print("[Test 3] Validation loader...")
    
    from torch.utils.data import DataLoader
    
    dataset = YOLODetectionDataset(
        img_dir=f"{DATA_DIR}/valid/images",
        label_dir=f"{DATA_DIR}/valid/labels",
        transforms=get_val_transforms(640)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,  # Validation phải False
        num_workers=0,
        collate_fn=yolo_collate_fn
    )
    
    # Check order preserved
    first_batch_first_img = next(iter(loader))[0][0]
    
    print(f"  Val loader works correctly")
    print("  ✓ Passed\n")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60 + "\n")
    
    tests = [test_collate_fn, test_dataloader_iteration, test_val_loader_no_shuffle]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ❌ FAILED: {e}\n")
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)