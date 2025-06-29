import numpy as np
import torch

from ImageProcessor import ImageProcessor
from LabelMapper import LabelMapper
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.processor = ImageProcessor() # Use the processor for postprocessing

    @staticmethod
    def calculate_iou_metric_single(prediction, ground_truth, num_classes=9):
        # prediction and ground_truth are expected to be (H, W) numpy arrays or tensors
        # containing class IDs (0-8)
        iou_per_class = []
        for cls in range(num_classes):
            pred_mask = (prediction == cls)
            gt_mask = (ground_truth == cls)

            intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
            union = torch.logical_or(pred_mask, gt_mask).sum().item()

            if union == 0:
                iou_per_class.append(1.0)  # If no pixels for this class in either, consider it perfect
            else:
                iou_per_class.append(intersection / union)
        return iou_per_class

    def calculate_iou_metric(self, prediction, ground_truth, num_classes=9):
        # Assicurati che siano tensori
        if not torch.is_tensor(prediction):
            prediction = torch.from_numpy(prediction)
        if not torch.is_tensor(ground_truth):
            ground_truth = torch.from_numpy(ground_truth)
        iou_per_class = []
        for cls in range(num_classes):
            pred_mask = (prediction == cls)
            gt_mask = (ground_truth == cls)

            intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
            union = torch.logical_or(pred_mask, gt_mask).sum().item()

            if union == 0:
                iou_per_class.append(1.0)
            else:
                iou_per_class.append(intersection / union)
        return iou_per_class

    def evaluate(self):
        self.model.eval()
        all_ious = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device) # Assume images are already preprocessed by DataLoader
                outputs = self.model(images)
                predictions = self.processor.postprocess(outputs) # Get uint8 (batch_size, rows, cols, 1)

                # Need to convert labels to the same format as predictions (batch_size, rows, cols)
                labels = labels.cpu().numpy() # If labels are already (batch_size, H, W)
                predictions = predictions.squeeze(-1).cpu().numpy() # Remove channel dim

                for i in range(images.shape[0]): # Iterate through batch
                    iou = self.calculate_iou_metric(predictions[i], labels[i])
                    all_ious.append(iou)

        # Average IoU for all images and classes
        mean_iou_per_class = np.mean(all_ious, axis=0)
        overall_mean_iou = np.mean(mean_iou_per_class)
        return overall_mean_iou, mean_iou_per_class


    def evaluate_classification_metrics(self, num_classes=9):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        self.model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for images, masks in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy().flatten())
                all_gts.append(masks.cpu().numpy().flatten())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_gts)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        return metrics


    def compare_random_label_and_prediction(self, val_dataset):
        import random
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from LabelMapper import LabelMapper

        self.model.eval()
        label_mapper = LabelMapper()
        for images, labels in self.test_loader:
            idx = random.randint(0, images.shape[0] - 1)
            image = images[idx:idx + 1].to(self.device)
            label = labels[idx].cpu().numpy()
            output = self.model(image)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            break

        if hasattr(val_dataset, 'indices'):
            original_idx = val_dataset.indices[idx]
            folder_path = val_dataset.dataset.samples[original_idx][0]
        else:
            folder_path = val_dataset.samples[idx][0]

        folder_name = os.path.basename(os.path.dirname(folder_path))

        def mask_to_rgb(mask):
            h, w = mask.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for cls in np.unique(mask):
                rgb[mask == cls] = label_mapper.class_id_to_rgb(cls)
            return rgb

        label_rgb = mask_to_rgb(label)
        pred_rgb = mask_to_rgb(pred)


        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Folder origine: {folder_name}")

        plt.subplot(1, 2, 1)
        plt.title("Label reale")
        plt.imshow(label_rgb.astype(np.uint8))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Predizione")
        plt.imshow(pred_rgb.astype(np.uint8))
        plt.axis('off')
        plt.show()