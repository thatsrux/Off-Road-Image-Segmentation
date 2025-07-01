import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.weight = weight  # class weights
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes, H, W) - output from the model
        # targets: (batch_size, H, W) - ground truth labels

        # Apply softmax to get probabilities for each class
        inputs = torch.softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        # Flatten label and prediction tensors for easier calculation
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1) # (batch_size, num_classes, H*W)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1) # (batch_size, num_classes, H*W)

        # Calculate Intersection and Union for each class
        intersection = (inputs * targets_one_hot).sum(dim=2)  # (batch_size, num_classes)
        total = (inputs + targets_one_hot).sum(dim=2)         # (batch_size, num_classes)

        # Calculate Dice score for each class
        dice = (2. * intersection + self.smooth) / (total + self.smooth) # (batch_size, num_classes)

        # Dice loss is 1 - Dice score
        loss = 1 - dice

        # Apply weights if provided
        if self.weight is not None:
            # Ensure weights are on the same device and match the number of classes
            if self.weight.device != loss.device:
                self.weight = self.weight.to(loss.device)
            if self.weight.size(0) != loss.size(1):
                 raise ValueError("Weight tensor must have the same number of elements as the number of classes.")

            # Expand weights to match the loss tensor shape for broadcasting
            weight = self.weight.unsqueeze(0).expand_as(loss) # (1, num_classes) -> (batch_size, num_classes)

            # Apply weights to the loss
            loss = loss * weight

        # Mean loss across classes, then mean across batch
        loss = loss.mean(dim=1).mean()

        return loss