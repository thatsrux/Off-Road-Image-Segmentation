from collections import Counter

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
        self.num_labels = 9  # Classi da 0 a 8 (0 = background, ignorata nella validazione)

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

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                predictions = self.image_processor.postprocess(outputs)
                batch_ious = []

                for idx_in_batch in range(predictions.shape[0]):
                    pred_single = predictions[idx_in_batch].cpu().numpy().squeeze()
                    label_single = labels[idx_in_batch].cpu().numpy().squeeze()
                    iou_scores = self.compute_all_iou(pred_single, label_single, self.num_labels)

                    for class_idx, iou in enumerate(iou_scores, start=1):  # classi da 1 a 8
                        if not np.isnan(iou):
                            print(f"Batch {batch_idx + 1}, Immagine {idx_in_batch + 1}, Classe {class_idx}: IoU = {iou:.4f}")
                        else:
                            print(f"Batch {batch_idx + 1}, Immagine {idx_in_batch + 1}, Classe {class_idx}: IoU = N/A")

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

    @staticmethod
    def compute_class_weights(dataloader, num_classes):
        from collections import Counter
        label_counts = Counter()
        total_pixels = 0
        for _, masks in dataloader:
            for mask in masks:
                pixels = mask.cpu().numpy().flatten()
                label_counts.update(pixels.tolist())
                total_pixels += len(pixels)

        weights = [0.0] * num_classes
        for i in range(num_classes):
            count = label_counts.get(i, 1e-6)
            weights[i] = total_pixels / (count * num_classes)
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def get_class_distribution(dataset):
        label_counts = Counter()
        for _, mask in dataset:
            label_counts.update(mask.numpy().flatten().tolist())
        return dict(sorted(label_counts.items()))

    def compute_iou(self, mask1, mask2, label):
        intersection = np.sum((mask1 == label) & (mask2 == label))
        union = np.sum((mask1 == label) | (mask2 == label))
        if union == 0:
            return np.nan
        return intersection / union

    def compute_all_iou(self, mask1, mask2, num_labels=8):
        iou_scores = np.zeros((num_labels))
        for label in range(num_labels):
            iou = self.compute_iou(mask1, mask2, label + 1)
            iou_scores[label] = iou
        return iou_scores