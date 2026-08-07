import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_path, is_best=False):
    # Atomic Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Lưu ra file tạm (.tmp) rồi mới đổi tên
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "model_best.pth")
        torch.save(checkpoint, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Đã tải checkpoint từ '{checkpoint_path}' (Epoch {checkpoint.get('epoch', 0)})")

    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("best_loss", float("inf")),
    )