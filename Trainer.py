import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ImageProcessor import ImageProcessor

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, focal_loss, dice_loss, early_stopping_patience=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.image_processor = ImageProcessor()
        self.early_stopping_patience = early_stopping_patience
        self.num_labels = 9  # classi da 0 a 8 (0 = background)
        self.focal_loss = focal_loss
        self.dice_loss = dice_loss

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_focal = 0.0
        running_dice = 0.0
        with tqdm(self.train_loader, desc="Training", leave=True) as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                loss = self.criterion(outputs, labels)
                focal = self.focal_loss(outputs, labels).item()
                dice = self.dice_loss(outputs, labels).item()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_focal += focal
                running_dice += dice
                # Aggiorna la barra con la loss media corrente
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = running_loss / len(self.train_loader)
        avg_focal = running_focal / len(self.train_loader)
        avg_dice = running_dice / len(self.train_loader)
        return avg_loss, avg_focal, avg_dice

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_focal = 0.0
        total_dice = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation", leave=True) as pbar:
                for batch_idx, (images, labels) in enumerate(pbar):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]
                    loss = self.criterion(outputs, labels)
                    focal = self.focal_loss(outputs, labels).item()
                    dice = self.dice_loss(outputs, labels).item()
                    total_loss += loss.item()
                    total_focal += focal
                    total_dice += dice
                    predictions = self.image_processor.postprocess(outputs)
                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels.cpu())
                    # Aggiorna la barra con la loss media corrente
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        avg_focal = total_focal / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        avg_dice = total_dice / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        avg_iou = 0.0
        if len(all_predictions) > 0 and len(all_labels) > 0:
            concatenated_predictions = torch.cat(all_predictions).numpy()
            concatenated_labels = torch.cat(all_labels).numpy()
            concatenated_predictions = concatenated_predictions.squeeze(axis=-1)
            iou_scores_all_classes = self.compute_all_iou(concatenated_predictions, concatenated_labels)
            avg_ious_per_class = np.nan_to_num(iou_scores_all_classes)
            avg_iou = np.nanmean(iou_scores_all_classes)
        else:
            avg_ious_per_class = np.zeros(self.num_labels)
        print(f"Mean IoU by class: {[f'{iou:.4f}' for iou in avg_ious_per_class]}")

        return avg_iou, avg_loss, avg_focal, avg_dice

    def run(self, num_epochs, model_save_path):
        best_val_iou = float('-inf')
        epochs_no_improve = 0
        train_losses = []
        val_losses = []
        train_focal_losses = []
        val_focal_losses = []
        train_dice_losses = []
        val_dice_losses = []

        # Inizializza lo scheduler
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',           # Monitoriamo la validation loss
            factor=0.5,           # Dimezza il LR
            patience=5,           # Dopo 5 epoche senza miglioramenti
        )

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss, train_focal, train_dice = self.train_epoch()
            train_losses.append(train_loss)
            train_focal_losses.append(train_focal)
            train_dice_losses.append(train_dice)
            print(f"Train Loss: {train_loss:.4f}")

            val_iou, val_loss, val_focal, val_dice = self.validate_epoch()
            val_losses.append(val_loss)
            val_focal_losses.append(val_focal)
            val_dice_losses.append(val_dice)
            print(f"Validation mIoU: {val_iou:.4f}, Validation Loss: {val_loss:.4f}")

            # Step del scheduler sulla validation loss
            scheduler.step(val_loss)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(self.model.state_dict(), model_save_path)
                print(f"New best model saved in {model_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping activated after {epoch + 1} epochs. Best mIoU: {best_val_iou:.4f}")
                break

        print(f"Best model (mIoU={best_val_iou:.4f}) saved in {model_save_path}")
        self.plot_losses(train_losses, val_losses, train_dice_losses, val_dice_losses, train_focal_losses, val_focal_losses)


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

    def compute_all_iou_batch(self, preds, labels, num_labels):
        if preds.ndim == labels.ndim + 1 and preds.shape[-1] == 1:
            preds = preds.squeeze(axis=-1) # Remove the last dimension if its size is 1
        iou_scores = np.zeros(num_labels)
        for label in range(num_labels):
            pred_mask = (preds == label)
            label_mask = (labels == label)
            intersection = np.logical_and(pred_mask, label_mask).sum()
            union = np.logical_or(pred_mask, label_mask).sum()
            if union == 0:
                iou_scores[label] = np.nan
            else:
                iou_scores[label] = intersection / union
        return iou_scores

    def plot_losses(self, train_losses, val_losses, train_dice_losses, val_dice_losses, train_focal_losses, val_focal_losses):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        # Plot Loss
        axs[0].plot(train_losses, label='Training Loss')
        axs[0].plot(val_losses, label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training vs Validation Loss')
        axs[0].legend()
        axs[0].grid(True)
        # Plot Dice Loss
        axs[1].plot(train_dice_losses, label='Train Dice Loss')
        axs[1].plot(val_dice_losses, label='Validation Dice Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Dice Loss')
        axs[1].set_title('Train vs Validation Dice Loss')
        axs[1].legend()
        axs[1].grid(True)
        # Plot Focal Loss
        axs[2].plot(train_focal_losses, label='Train Focal Loss')
        axs[2].plot(val_focal_losses, label='Validation Focal Loss')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Focal Loss')
        axs[2].set_title('Train vs Validation Focal Loss')
        axs[2].legend()
        axs[2].grid(True)
        plt.tight_layout()
        plt.show()
