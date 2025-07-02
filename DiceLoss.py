import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) - logits (non softmaxati)
        targets: (N, H, W) - valori interi da 0 a C-1
        """
        inputs = F.softmax(inputs, dim=1)  # Convert logits to probabilities

        total_dice = 0.0
        valid_classes = 0

        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index:
                continue

            # Create binary masks for the current class
            inputs_class = inputs[:, class_idx, :, :]  # (N, H, W)
            targets_class = (targets == class_idx).float()  # (N, H, W)

            intersection = torch.sum(inputs_class * targets_class)
            union = torch.sum(inputs_class) + torch.sum(targets_class)

            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_dice += 1.0 - dice_score  # Dice loss
            valid_classes += 1

        return total_dice / valid_classes if valid_classes > 0 else torch.tensor(0.0, device=inputs.device)
