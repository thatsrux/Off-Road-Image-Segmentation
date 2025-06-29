from Evaluator import Evaluator
from ImageProcessor import ImageProcessor
import torch

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
                labels_device = labels

                current_batch_iou_sum = 0.0
                for i in range(predictions.shape[0]):
                    pred_single = predictions[i].squeeze(-1)
                    label_single = labels_device[i]
                    iou_per_class = Evaluator.calculate_iou_metric_single(
                        pred_single, label_single, num_classes=9
                    )
                    iou_per_class_tensor = torch.tensor(iou_per_class, device=self.device)
                    current_batch_iou_sum += torch.mean(iou_per_class_tensor.float()).item()

                batch_iou = current_batch_iou_sum
                total_iou += batch_iou
                num_batches += 1
                print(f"Batch {batch_idx + 1}/{len(self.val_loader)}, mIoU: {batch_iou:.4f}")

        return total_iou / num_batches if num_batches > 0 else 0.0


    def run(self, num_epochs, model_save_path):
        best_val_iou = float('-inf')
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
        print(f"Modello migliore (mIoU={best_val_iou:.4f}) salvato in {model_save_path}")
