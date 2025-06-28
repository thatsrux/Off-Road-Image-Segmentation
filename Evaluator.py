import numpy as np
import torch

from ImageProcessor import ImageProcessor


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

    @staticmethod
    def build_prediction_image(predictions_batch):

        if torch.is_tensor(predictions_batch):
            predictions_batch = predictions_batch.cpu().numpy()
        # Impila le immagini verticalmente
        return np.vstack(predictions_batch)