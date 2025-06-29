from SegmentationModel_parts import *
import torch # Import added for clarity if not already present globally

class SegmentationModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SegmentationModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Canali ridotti per una rete più leggera
        self.inc = (DoubleConv(n_channels, 32))      # Da 64 a 32
        self.down1 = (Down(32, 64))                  # Da 128 a 64
        self.down2 = (Down(64, 128))                 # Da 256 a 128
        self.down3 = (Down(128, 256))                # Da 512 a 256
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))      # Da 1024 a 512 (o 256 se bilinear)
        self.up1 = (Up(512, 256 // factor, bilinear)) # Da 1024 a 512 in ingresso, output 256 (o 128 se bilinear)
        self.up2 = (Up(256, 128 // factor, bilinear)) # Da 512 a 256 in ingresso, output 128 (o 64 se bilinear)
        self.up3 = (Up(128, 64 // factor, bilinear))  # Da 256 a 128 in ingresso, output 64 (o 32 se bilinear)
        self.up4 = (Up(64, 32, bilinear))            # Da 128 a 64 in ingresso, output 32
        self.outc = (OutConv(32, n_classes))         # Output layer

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        # Mantiene il checkpointing per ottimizzazione della memoria
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)