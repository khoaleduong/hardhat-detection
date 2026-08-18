import torch
from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def move_targets_to_device(targets, device):
    return [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]


def train_one_epoch(model, loader, optimizer, epoch, device, writer=None, scaler=None):
    model.train()
    amp_enabled = torch.device(device).type == "cuda"
    total_loss = 0.0
    loss_meter = {}
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]")

    for step, (images, targets) in enumerate(pbar, 1):
        # Move data to device
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)

        # Forward
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        # Catch NaN / Inf early
        if not torch.isfinite(loss):  # type: ignore
            loss_details = {name: value.item() for name, value in loss_dict.items()}
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}. Losses: {loss_details}")  # type: ignore

        # Backward
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()  # type: ignore
            optimizer.step()

        # Metrics
        loss_value = loss.item()  # type: ignore
        total_loss += loss_value
        for name, value in loss_dict.items():
            loss_meter[name] = loss_meter.get(name, 0.0) + value.item()

        avg_loss = total_loss / step
        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    num_batches = len(loader)
    results = {"loss": total_loss / num_batches}
    for name, value in loss_meter.items():
        results[name] = value / num_batches

    if writer is not None:
        for name, value in results.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]["lr"], epoch)

    return results


@torch.no_grad()
def validate_loss(model, loader, device, epoch, writer=None):
    """Compute Faster R-CNN validation loss. Note: TorchVision Faster R-CNN only returns loss_dict while the model is in training mode."""
    model.train()
    amp_enabled = torch.device(device).type == "cuda"
    total_loss = 0.0
    loss_meter = {}
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Valid]")

    for step, (images, targets) in enumerate(pbar, 1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        loss_value = loss.item()  # type: ignore
        total_loss += loss_value
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


@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    model.eval()
    amp_enabled = torch.device(device).type == "cuda"
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    pbar = tqdm(loader, desc="Evaluation")

    for images, targets in pbar:
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(images)

        # TorchMetrics works on CPU
        outputs = [{key: value.cpu() for key, value in output.items()} for output in outputs]
        targets = [{key: value.cpu() if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]
        metric.update(outputs, targets)

    results = metric.compute()

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"{'mAP@50:95':<20}: {results['map'].item():.4f}")
    print(f"{'mAP@50':<20}: {results['map_50'].item():.4f}")
    print(f"{'mAP@75':<20}: {results['map_75'].item():.4f}")
    print(f"{'mAR@1':<20}: {results['mar_1'].item():.4f}")
    print(f"{'mAR@10':<20}: {results['mar_10'].item():.4f}")
    print(f"{'mAR@100':<20}: {results['mar_100'].item():.4f}")

    if results["map_per_class"].numel() > 0:
        print("\nAP Per Class")
        print("-" * 60)
        aps = results["map_per_class"]
        class_ids = results["classes"].tolist()
        for class_id, ap in zip(class_ids, aps):
            if class_names is not None and class_id < len(class_names):
                name = class_names[class_id]
            else:
                name = f"Class {class_id}"
            print(f"{name:<20}: {ap.item():.4f}")

    print("=" * 60)
    return results