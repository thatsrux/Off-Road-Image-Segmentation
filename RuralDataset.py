import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from LabelMapper import LabelMapper
from tqdm.auto import tqdm

class RuralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapper = LabelMapper()
        self.samples = []
        self.classes_per_sample = []

        # Wrap the folder iteration with tqdm
        for folder_name in tqdm(os.listdir(root_dir), desc=f"Loading data from {root_dir}"):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                rgb_path = os.path.join(folder_path, 'rgb.jpg')
                labels_path = os.path.join(folder_path, 'labels.png')
                if os.path.exists(rgb_path) and os.path.exists(labels_path):
                    label_image = Image.open(labels_path).convert("RGB")
                    label_np = np.array(label_image)
                    # Vettorizzazione: shape (H, W, 3) -> (H*W, 3)
                    flat_label = label_np.reshape(-1, 3)
                    # Applica la mappatura a tutti i pixel
                    class_ids = np.array([self.label_mapper.rgb_to_class_id(tuple(rgb)) for rgb in flat_label])
                    class_id_mask = class_ids.reshape(label_np.shape[:2])
                    unique_classes = np.unique(class_id_mask)
                    self.classes_per_sample.append(unique_classes)
                    if np.any(np.isin(unique_classes, [6, 7, 8])):
                        self.samples.append((rgb_path, labels_path))
                        self.samples.append((rgb_path, labels_path))
                    else:
                        self.samples.append((rgb_path, labels_path))
                else:
                    print(f"Warning: Missing rgb.jpg or labels.png in {folder_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, labels_path = self.samples[idx]
        image = Image.open(rgb_path).convert("RGB")
        label_image = Image.open(labels_path).convert("RGB")
        resize_size = None
        if self.transform is not None:
            for t in getattr(self.transform, 'transforms', []):
                if t.__class__.__name__ == 'Resize':
                    resize_size = t.size if hasattr(t, 'size') else t.args[0]
                    break
        if resize_size:
            image = image.resize((resize_size[1], resize_size[0]), Image.BILINEAR)
            label_image = label_image.resize((resize_size[1], resize_size[0]), Image.NEAREST)
        label_np = np.array(label_image)
        flat_label = label_np.reshape(-1, 3)
        class_ids = np.array([self.label_mapper.rgb_to_class_id(tuple(rgb)) for rgb in flat_label])
        class_id_mask = class_ids.reshape(label_np.shape[:2])
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.from_numpy(class_id_mask).long()
        return image, label_tensor