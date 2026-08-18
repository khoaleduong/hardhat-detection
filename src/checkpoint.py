from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_loss,
    save_path,
    scaler=None,
):
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    if scaler is not None:
        checkpoint["scaler_state_dict"] = (
            scaler.state_dict()
        )

    # Atomic replacement
    tmp_path = save_path.with_suffix(
        save_path.suffix + ".tmp"
    )

    torch.save(checkpoint, tmp_path)
    tmp_path.replace(save_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    scaler=None,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if (
        scaler is not None
        and "scaler_state_dict" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get(
        "best_loss",
        float("inf"),
    )

    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(epoch {epoch})"
    )

    return epoch, best_loss
