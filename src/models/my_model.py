import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# num_classes: background + helmet + head + person
# default trainable_backbone_layers: 5
def build_model(num_classes=4, trainable_backbone_layers=3):
  weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

  # Build model
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
      weights=weights,
      trainable_backbone_layers=trainable_backbone_layers
  )

  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": f"{total / 1e6:.1f}M",
        "trainable": f"{trainable / 1e6:.1f}M",
        "frozen": f"{frozen / 1e6:.1f}M",
    }