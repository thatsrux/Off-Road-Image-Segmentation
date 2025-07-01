import torch
import numpy as np
from ImageProcessor import ImageProcessor


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, early_stopping_patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.image_processor = ImageProcessor()
        self.early_stopping_patience = early_stopping_patience

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_iou = 0.0
        num_batches = 0
        num_labels = 8
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                predictions = self.image_processor.postprocess(outputs)
                batch_ious = []

                for i in range(predictions.shape[0]):
                    pred_single = predictions[i].cpu().numpy().squeeze()
                    label_single = labels[i].cpu().numpy().squeeze()
                    iou_scores = np.zeros(num_labels)
                    for label in range(num_labels):
                        intersection = np.sum((pred_single == (label + 1)) & (label_single == (label + 1)))
                        union = np.sum((pred_single == (label + 1)) | (label_single == (label + 1)))
                        if union == 0:
                            iou = np.nan
                        else:
                            iou = intersection / union
                        iou_scores[label] = iou
                    mean_iou = np.nanmean(iou_scores)
                    batch_ious.append(mean_iou)

                batch_iou = np.nanmean(batch_ious)
                total_iou += batch_iou
                num_batches += 1
                print(f"Batch {batch_idx + 1}/{len(self.val_loader)}, mIoU: {batch_iou:.4f}")

        return total_iou / num_batches if num_batches > 0 else 0.0

    def run(self, num_epochs, model_save_path):
        best_val_iou = float('-inf')
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            val_iou = self.validate_epoch()
            print(f"Validation mIoU: {val_iou:.4f}")
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Nuovo modello migliore salvato in {model_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Nessun miglioramento per {epochs_no_improve} epoche.")
            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping attivato dopo {epoch + 1} epoche. Miglior mIoU: {best_val_iou:.4f}")
                break
        print(f"Modello migliore (mIoU={best_val_iou:.4f}) salvato in {model_save_path}")
