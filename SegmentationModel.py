from torch import nn


class SegmentationModel(nn.Module):
    def __init__(self, num_classes=9):
        super(SegmentationModel, self).__init__()
        # Esempio semplificato: dovrai definire una vera architettura
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x is (batch_size, 3, rows, cols)
        x = self.relu(self.conv1(x))
        # Final output for pixel classification (batch_size, num_classes, rows, cols)
        return self.output_conv(x)