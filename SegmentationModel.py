import segmentation_models_pytorch as smp
import torch.nn as nn


class SegmentationModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # Usa Unet con encoder ResNet34 pre-addestrato su ImageNet
        self.model = smp.Unet(
            encoder_name="resnet34",
            in_channels=3,
            classes=n_class,
        )

    def forward(self, x):
        return self.model(x)