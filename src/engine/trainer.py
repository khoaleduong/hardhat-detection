import time
from typing import Dict
import torch
from tqdm.auto import tqdm

def train_one_epoch(model, loader, optimizer, epoch, device, writer=None):
    model.train()

    total_loss = 0.0
    loss_meter = {}

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for step, (images, targets) in enumerate(pbar):
        # 1. Chuyển data lên device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. Forward pass
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        # 3. Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # type: ignore
        optimizer.step()

        # 4. Cộng dồn loss (dùng item() để ngắt computation graph)
        with torch.no_grad():
            loss_val = loss.item() # type: ignore
            total_loss += loss_val

            for name, value in loss_dict.items():
                loss_meter[name] = loss_meter.get(name, 0.0) + value.item()

            # 5. Cập nhật progress bar
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    # 6. Tính toán loss trung bình cho cả Epoch
    num_batches = len(loader)
    results = {"loss": total_loss / num_batches}

    for name, value in loss_meter.items():
        results[name] = value / num_batches

    # 7. Ghi log vào TensorBoard
    if writer is not None:
        for name, value in results.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]["lr"], epoch)

    return results