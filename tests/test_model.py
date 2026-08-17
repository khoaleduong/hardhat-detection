import torch
from src.faster_rcnn_model import build_model
import pytest


def test_model_output_classes():
    model = build_model(num_classes=4, pretrained=False, trainable_backbone_layers=3)
    assert model.roi_heads.box_predictor.cls_score.out_features == 4 # type: ignore

@pytest.mark.slow
def test_model_train_forward():
    model = build_model(num_classes=4, pretrained=False)
    model.train()

    images = [torch.rand(3, 128, 160)]
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 80.0, 100.0]], dtype=torch.float32), "labels": torch.tensor([1], dtype=torch.int64)}]

    losses = model(images, targets)

    assert "loss_classifier" in losses
    assert "loss_box_reg" in losses
    assert "loss_objectness" in losses
    assert "loss_rpn_box_reg" in losses

    for loss in losses.values():
        assert torch.isfinite(loss)

def test_model_eval_forward():
    model = build_model(num_classes=4, pretrained=False)
    model.eval()

    images = [torch.rand(3, 128, 160)]

    with torch.no_grad():
        outputs = model(images)

    assert len(outputs) == 1
    output = outputs[0]
    assert "boxes" in output
    assert "labels" in output
    assert "scores" in output
    assert output["boxes"].ndim == 2
    assert output["boxes"].shape[1] == 4