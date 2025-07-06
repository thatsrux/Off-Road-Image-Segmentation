import numpy as np
import torch

from LabelMapper import LabelMapper


class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    @staticmethod
    def calculate_iou_metric_single(prediction, ground_truth, num_classes=9):
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

    def calculate_iou_metric(self, prediction, ground_truth, num_classes=9):
        def compute_iou(mask1, mask2, label):
            intersection = np.sum((mask1 == label) & (mask2 == label))
            union = np.sum((mask1 == label) | (mask2 == label))
            if union == 0:
                return np.nan
            return intersection / union
        def compute_all_iou(mask1, mask2, num_labels=8):
            iou_scores = np.zeros((num_labels))
            for label in range(num_labels):
                iou = compute_iou(mask1, mask2, label+1)
                iou_scores[label] = iou
            return iou_scores
        if not isinstance(prediction, np.ndarray):
            prediction = prediction.cpu().numpy() if hasattr(prediction, 'cpu') else np.array(prediction)
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = ground_truth.cpu().numpy() if hasattr(ground_truth, 'cpu') else np.array(ground_truth)
        return compute_all_iou(prediction, ground_truth, num_labels=num_classes-1)

    def evaluate(self):
        self.model.eval()
        all_ious = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                for i in range(images.shape[0]):
                    iou = self.calculate_iou_metric(predictions[i], labels[i])
                    all_ious.append(iou)
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
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
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
        import os
        import matplotlib.pyplot as plt
        self.model.eval()
        label_mapper = LabelMapper()
        for images, labels in self.test_loader:
            idx = random.randint(0, images.shape[0] - 1)
            image = images[idx:idx + 1].to(self.device)
            label = labels[idx].cpu().numpy()
            output = self.model(image)
            if isinstance(output, dict):
                output = output["out"]
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

    def predict_from_folder(self, folder_number, data_root='train'):
        import os
        import matplotlib.pyplot as plt
        from PIL import Image
        from torchvision import transforms
        folder_name = f"{int(folder_number):04d}"
        folder_path = os.path.join(data_root, folder_name)
        rgb_path = os.path.join(folder_path, 'rgb.jpg')
        label_path = os.path.join(folder_path, 'labels.png')
        if not (os.path.exists(rgb_path) and os.path.exists(label_path)):
            print(f"Immagini non trovate in {folder_path}")
            return
        image = Image.open(rgb_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")
        label_np = np.array(label_image)
        label_mapper = LabelMapper()
        class_id_mask = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)
        for r in range(label_np.shape[0]):
            for c in range(label_np.shape[1]):
                pixel_rgb = tuple(label_np[r, c])
                class_id_mask[r, c] = label_mapper.rgb_to_class_id(pixel_rgb)
        class_id_mask = np.array(Image.fromarray(class_id_mask).resize((512, 272), resample=Image.NEAREST))
        val_transform = transforms.Compose([
            transforms.Resize((272, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = val_transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, dict):
                output = output["out"]
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        def id_to_rgb_mask(id_mask, id_to_color_map):
            h, w = id_mask.shape
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id, color_rgb in id_to_color_map.items():
                rgb_mask[id_mask == class_id] = color_rgb
            return rgb_mask

        id_to_color = label_mapper.class_id_to_color
        true_label_rgb = id_to_rgb_mask(class_id_mask, id_to_color)
        pred_label_rgb = id_to_rgb_mask(pred_mask, id_to_color)
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title('Immagine RGB')
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Label reale')
        plt.imshow(true_label_rgb)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('Label predetta')
        plt.imshow(pred_label_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        correct = (pred_mask == class_id_mask).sum()
        total = class_id_mask.size
        print(f"Pixel classificati correttamente: {correct} / {total} ({correct / total:.2%})")
        iou_scores = self.compute_all_iou(pred_mask, class_id_mask, num_labels=8)
        mean_iou = np.nanmean(iou_scores)
        print(f"IoU medio sull'immagine: {mean_iou:.4f}")
        print(f"IoU per classe: {iou_scores}")

    def predict_from_all_folders(self, data_root='test'):
        import os
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        import torch
        import matplotlib.pyplot as plt
        label_mapper = LabelMapper()
        all_iou_scores = []
        all_accuracies = []
        folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
        folders.sort()
        for folder_name in folders:
            folder_path = os.path.join(data_root, folder_name)
            rgb_path = os.path.join(folder_path, 'rgb.jpg')
            label_path = os.path.join(folder_path, 'labels.png')
            if not (os.path.exists(rgb_path) and os.path.exists(label_path)):
                print(f"Immagini non trovate in {folder_path}")
                continue
            image = Image.open(rgb_path).convert("RGB")
            label_image = Image.open(label_path).convert("RGB")
            label_np = np.array(label_image)
            class_id_mask = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)
            for r in range(label_np.shape[0]):
                for c in range(label_np.shape[1]):
                    pixel_rgb = tuple(label_np[r, c])
                    class_id_mask[r, c] = label_mapper.rgb_to_class_id(pixel_rgb)
            class_id_mask = np.array(Image.fromarray(class_id_mask).resize((512, 272), resample=Image.NEAREST))
            val_transform = transforms.Compose([
                transforms.Resize((272, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = val_transform(image).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                if isinstance(output, dict):
                    output = output["out"]
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            correct = (pred_mask == class_id_mask).sum()
            total = class_id_mask.size
            accuracy = correct / total
            iou_scores = self.compute_all_iou(pred_mask, class_id_mask, num_labels=8)
            mean_iou = np.nanmean(iou_scores)
            all_iou_scores.append(iou_scores)
            all_accuracies.append(accuracy)
            print(f"Cartella: {folder_name}")
            print(f"  Pixel classificati correttamente: {correct} / {total} ({accuracy:.2%})")
            print(f"  IoU medio sull'immagine: {mean_iou:.4f}")
            print(f"  IoU per classe: {iou_scores}")

            # --- PLOT ---
            def id_to_rgb_mask(id_mask, id_to_color_map):
                h, w = id_mask.shape
                rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for class_id, color_rgb in id_to_color_map.items():
                    rgb_mask[id_mask == class_id] = color_rgb
                return rgb_mask

            id_to_color = label_mapper.class_id_to_color
            true_label_rgb = id_to_rgb_mask(class_id_mask, id_to_color)
            pred_label_rgb = id_to_rgb_mask(pred_mask, id_to_color)
            plt.figure(figsize=(18, 6))
            plt.suptitle(f"Risultati cartella: {folder_name}")
            plt.subplot(1, 3, 1)
            plt.title('Immagine RGB')
            plt.imshow(image)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.title('Label reale')
            plt.imshow(true_label_rgb)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title('Label predetta')
            plt.imshow(pred_label_rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        if all_iou_scores:
            all_iou_scores = np.array(all_iou_scores)
            mean_iou_per_class = np.nanmean(all_iou_scores, axis=0)
            mean_iou_total = np.nanmean(all_iou_scores)
            mean_accuracy = np.mean(all_accuracies)
            print("\n--- Risultati medi su tutte le cartelle ---")
            print(f"Accuratezza media: {mean_accuracy:.2%}")
            print(f"IoU medio totale: {mean_iou_total:.4f}")
            print(f"IoU medio per classe: {mean_iou_per_class}")
        else:
            print("Nessuna cartella valida trovata.")
