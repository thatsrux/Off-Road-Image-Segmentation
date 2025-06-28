import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from LabelMapper import LabelMapper


class RuralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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

        # Load RGB image
        image = Image.open(rgb_path).convert("RGB")
        # Load labels image - ensure it's loaded as RGB to read color values
        label_image = Image.open(labels_path).convert("RGB")

        # Convert label_image from RGB to class IDs
        # This will create a numpy array (H, W) where each pixel is a class ID (0-8)
        label_np = np.array(label_image) # (H, W, 3)
        # Create an empty array for the class IDs
        class_id_mask = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)

        for r in range(label_np.shape[0]):
            for c in range(label_np.shape[1]):
                pixel_rgb = tuple(label_np[r, c])
                class_id_mask[r, c] = self.label_mapper.rgb_to_class_id(pixel_rgb)

        # Apply transformations if any
        if self.transform:
            # Note: For transformations, you might need to ensure they handle both image and mask
            # For segmentation, often transforms are applied sequentially or a custom transform
            # is used that modifies both. Here, a simple example where transforms are applied
            # independently, which might not be ideal for operations like resizing/cropping.
            image = self.transform(image)
            # For the label mask, if transform includes resize, it must be applied with
            # interpolation that preserves discrete labels (e.g., nearest neighbor)
            # Example if 'transform' also handles label_image resizing:
            # label_image_transformed = self.transform_labels(Image.fromarray(label_np))
            # You would then re-process label_image_transformed to class_id_mask if it's RGB
            # Or, directly transform the class_id_mask using a suitable interpolation
            # For simplicity, assuming transforms are for images only or handle masks correctly
            # If your transform requires PIL Image input, convert class_id_mask back to PIL
            # and then convert back to numpy/tensor after transform.
            # A more robust approach might be to transform both image and mask together.

        # Convert to PyTorch tensors
        # Image: (C, H, W) float, normalized
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        # Label: (H, W) long (for CrossEntropyLoss)
        label_tensor = torch.from_numpy(class_id_mask).long()

        return image_tensor, label_tensor