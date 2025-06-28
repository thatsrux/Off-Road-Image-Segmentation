import numpy as np
import torch

from Evaluator import Evaluator
from ImageProcessor import ImageProcessor


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.image_processor = ImageProcessor()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_iou = 0.0
        num_batches = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                predictions = self.image_processor.postprocess(outputs.cpu())
                labels_cpu = labels.cpu()

                current_batch_iou_sum = 0.0
                for i in range(predictions.shape[0]):
                    pred_single = predictions[i].squeeze(-1)
                    label_single = labels_cpu[i]
                    iou_per_class = Evaluator.calculate_iou_metric_single(pred_single, label_single, num_classes=9)
                    current_batch_iou_sum += np.mean(iou_per_class)

                total_iou += current_batch_iou_sum
                num_batches += 1

        return total_iou / num_batches if num_batches > 0 else 0.0

    def run(self, num_epochs, model_save_path="best_model.pth"):
        best_val_iou = -1.0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_iou = self.validate_epoch()
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Model saved! Best Val IoU: {best_val_iou:.4f}")