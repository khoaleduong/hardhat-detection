from typing import Dict
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def validate(model, loader, device, epoch: int, writer=None) -> Dict[str, float]:
    # Faster R-CNN trả về loss_dict ở train mode
    model.train()

    total_loss = 0.0
    loss_meter = {}
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Valid]")

    # 2. KHÓA (Freeze) BatchNorm để không bị cập nhật running statistics trên tập Validation
    for module in model.modules():
      if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
          module.eval()

    for step, (images, targets) in enumerate(pbar, 1):
      images = [img.to(device) for img in images]
      targets = [
          {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
          for t in targets
      ]

      loss_dict = model(images, targets)
      loss = sum(loss_dict.values())

      total_loss += loss.item() # type: ignore
      for name, value in loss_dict.items():
          loss_meter[name] = loss_meter.get(name, 0.0) + value.item()

      pbar.set_postfix(loss=f"{total_loss / step:.4f}")

    num_batches = len(loader)
    results = {"loss": total_loss / num_batches}
    for name, value in loss_meter.items():
      results[name] = value / num_batches

    if writer is not None:
      for name, value in results.items():
          writer.add_scalar(f"Valid/{name}", value, epoch)

    return results