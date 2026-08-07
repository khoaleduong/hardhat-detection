# test_full_pipeline.py
import torch
from src.dataset.dataloader import build_dataloader, yolo_collate_fn

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = "/Users/leduongkhoa/hardhat_detection/data"
# ==========================================

def test_full_pipeline():
    """Test toàn bộ pipeline"""
    print("[Full Pipeline Test]")
    
    # Build dataloaders
    train_loader, val_loader = build_dataloader(
        data_dir=DATA_DIR,
        img_size=640,
        batch_size=8,
        num_workers=0
    )
    
    # Test train loader
    print("\nTrain Loader:")
    for i, (images, targets) in enumerate(train_loader):
        if i >= 2:
            break
        
        print(f"  Batch {i}:")
        print(f"    Images: {images.shape}")
        print(f"    Targets: {len(targets)} dicts")
        
        # Check normalization
        print(f"    Image range: [{images.min():.2f}, {images.max():.2f}]")
        
        # Check boxes
        total_boxes = sum(len(t['boxes']) for t in targets)
        print(f"    Total boxes: {total_boxes}")
    
    # Test val loader
    print("\nValidation Loader:")
    for i, (images, targets) in enumerate(val_loader):
        if i >= 2:
            break
        
        print(f"  Batch {i}:")
        print(f"    Images: {images.shape}")
        print(f"    Targets: {len(targets)} dicts")
    
    print("\n✓ Full pipeline test passed!")

if __name__ == "__main__":
    test_full_pipeline()