import os
import time
import shutil
import torch
import torch.optim as optim

from src.engine.trainer import train_one_epoch
from src.engine.evaluator import validate
from src.utils.checkpoint import save_checkpoint

def train(
    model, 
    train_loader, 
    valid_loader, 
    optimizer, 
    scheduler, 
    device, 
    num_epochs, 
    start_epoch=1,          # Thêm start_epoch hỗ trợ resume
    best_loss=float("inf"), # Truyền best_loss cũ vào nếu resume
    writer=None, 
    save_dir="checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    history = {"train": [], "valid": []}
    
    print("=" * 70 + f"\nStart Training from Epoch {start_epoch} to {num_epochs}\n" + "=" * 70)

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        
        # 1. Train & Validate
        train_metrics = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device, epoch=epoch, writer=writer
        )
        valid_metrics = validate(
            model=model, loader=valid_loader, device=device, epoch=epoch, writer=writer
        )

        history["train"].append(train_metrics)
        history["valid"].append(valid_metrics)

        # 2. Update Scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_metrics["loss"])
            else:
                scheduler.step()

        # 3. Luồng Save Checkpoint (Tránh ghi đĩa 2 lần)
        val_loss = valid_metrics["loss"]
        is_best = val_loss < best_loss
        if is_best: 
            best_loss = val_loss

        last_path = os.path.join(save_dir, "last_model.pth")
        best_path = os.path.join(save_dir, "best_model.pth")

        # Luôn lưu file checkpoint mới nhất
        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, last_path)
        
        # Nếu đạt Best thì chỉ COPY file sang best_model, nhanh hơn SAVE
        if is_best:
            shutil.copyfile(last_path, best_path)
            print("Best model updated.")

        # 4. Log thông tin ra Terminal
        elapsed = time.time() - start_time
        print(f"[{epoch:03d}/{num_epochs:03d}] Train Loss: {train_metrics['loss']:.4f} | Valid Loss: {val_loss:.4f} | Time: {elapsed:.1f}s\n" + "-" * 70)

    print(f"\nTraining Finished.\nBest Validation Loss: {best_loss:.4f}")
    if writer:
        writer.close()

    return history