import torch
import numpy as np
from collections import Counter
from ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt


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
        self.num_labels = 9  # classi da 0 a 8 (0 = background)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_ious = np.zeros(self.num_labels)
        num_batches = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            # Se outputs è un dizionario, prendi la chiave 'out'
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            # Calcolo IoU per batch
            predictions = self.image_processor.postprocess(outputs)
            batch_ious = np.zeros(self.num_labels)
            for idx_in_batch in range(predictions.shape[0]):
                pred_single = predictions[idx_in_batch].cpu().numpy().squeeze()
                label_single = labels[idx_in_batch].cpu().numpy().squeeze()
                iou_scores = self.compute_all_iou(pred_single, label_single, self.num_labels)
                batch_ious += np.nan_to_num(iou_scores)
            total_ious += batch_ious / predictions.shape[0]
            num_batches += 1
            print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        avg_ious = total_ious / num_batches if num_batches > 0 else np.zeros(self.num_labels)
        print(f"{[f'{iou:.4f}' for iou in avg_ious]}")
        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_iou = 0.0
        total_loss = 0.0
        num_batches = 0
        total_ious_per_class = np.zeros(self.num_labels)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                # Se outputs è un dizionario, prendi la chiave 'out'
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                predictions = self.image_processor.postprocess(outputs)
                batch_ious = []
                batch_ious_per_class = np.zeros(self.num_labels)
                for idx_in_batch in range(predictions.shape[0]):
                    pred_single = predictions[idx_in_batch].cpu().numpy().squeeze()
                    label_single = labels[idx_in_batch].cpu().numpy().squeeze()
                    iou_scores = self.compute_all_iou(pred_single, label_single, self.num_labels)
                    batch_ious.append(np.nanmean(iou_scores))
                    batch_ious_per_class += np.nan_to_num(iou_scores)
                batch_iou = np.nanmean(batch_ious)
                total_iou += batch_iou
                # Media IoU per classe per batch
                total_ious_per_class += batch_ious_per_class / predictions.shape[0]
                num_batches += 1
                print(f"Batch {batch_idx + 1}/{len(self.val_loader)}, mIoU: {batch_iou:.4f}, Loss: {loss.item():.4f}")

        avg_ious_per_class = total_ious_per_class / num_batches if num_batches > 0 else np.zeros(self.num_labels)
        print(f"IoU medio per classe (validation): {[f'{iou:.4f}' for iou in avg_ious_per_class]}")
        avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_iou, avg_loss

    def run(self, num_epochs, model_save_path):
        best_val_iou = float('-inf')
        epochs_no_improve = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            val_iou, val_loss = self.validate_epoch()
            val_losses.append(val_loss)
            print(f"Validation mIoU: {val_iou:.4f}, Validation Loss: {val_loss:.4f}")

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
        self.plot_losses(train_losses, val_losses)

    @staticmethod
    def compute_class_weights(dataloader, num_classes):
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

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
