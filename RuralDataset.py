import os
import numpy as np
import torch
from IPython.core.display_functions import display
from PIL import Image
from torch.utils.data import Dataset

from LabelMapper import LabelMapper


class RuralDataset(Dataset):
    def __init__(self, root_dir, transform=None, device=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapper = LabelMapper()
        self.samples = [] # List to store paths to each sample folder
        # Populate samples list
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                rgb_path = os.path.join(folder_path, 'rgb.jpg')
                labels_path = os.path.join(folder_path, 'labels.png')
                if os.path.exists(rgb_path) and os.path.exists(labels_path):
                    self.samples.append((rgb_path, labels_path))
                else:
                    print(f"Warning: Missing rgb.jpg or labels.png in {folder_path}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, labels_path = self.samples[idx]
        image = Image.open(rgb_path).convert("RGB")
        label_image = Image.open(labels_path).convert("RGB")
        label_np = np.array(label_image)
        class_id_mask = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)
        for r in range(label_np.shape[0]):
            for c in range(label_np.shape[1]):
                pixel_rgb = tuple(label_np[r, c])
                class_id_mask[r, c] = self.label_mapper.rgb_to_class_id(pixel_rgb)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.from_numpy(class_id_mask).long()
        image = image
        return image, label_tensor