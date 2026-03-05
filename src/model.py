import torch.nn as nn
from torchvision import models


def build_vgg16(num_classes: int) -> nn.Module:
    """
    Load pretrained VGG-16, freeze the feature extractor, and replace the
    final classifier layer to match the target number of classes.

    Only the classifier parameters will have requires_grad=True.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    return model
