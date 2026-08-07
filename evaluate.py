import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )

    for images, targets in tqdm(loader, desc="Evaluation"):
      images = [img.to(device) for img in images]
      outputs = model(images)

      outputs = [
          {k: v.cpu() for k, v in output.items()} for output in outputs
      ]
      targets = [
          {k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()}
          for t in targets
      ]
      metric.update(outputs, targets)

    results = metric.compute()

    print("=" * 60 + "\nEvaluation Results\n" + "=" * 60)
    print(f"{'mAP@50:95':<20}: {results['map'].item():.4f}")
    print(f"{'mAP@50':<20}: {results['map_50'].item():.4f}")
    print(f"{'mAP@75':<20}: {results['map_75'].item():.4f}")
    print(f"{'mAR@1':<20}: {results['mar_1'].item():.4f}")
    print(f"{'mAR@10':<20}: {results['mar_10'].item():.4f}")
    print(f"{'mAR@100':<20}: {results['mar_100'].item():.4f}")

    if "map_per_class" in results and results["map_per_class"].numel() > 0:
      print("\nAP Per Class\n" + "-" * 60)
      aps = results["map_per_class"]
      class_ids = (
          results["classes"].tolist()
          if "classes" in results
          else list(range(len(aps)))
      )

      for c_id, ap in zip(class_ids, aps):
          name = (
              class_names[c_id]
              if class_names and c_id < len(class_names)
              else f"Class {c_id}"
          )
          print(f"{name:<20}: {ap.item():.4f}")

    print("=" * 60)
    return results