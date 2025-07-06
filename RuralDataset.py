import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LabelMapper import LabelMapper


class RuralDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapper = LabelMapper()
        self.augment = augment
        self.samples = []

        # Costruzione lookup table RGB -> class_id (shape: [256, 256, 256])
        self.rgb_to_id = np.full((256, 256, 256), 0, dtype=np.uint8)  # Default = background (0)
        for rgb, class_id in self.label_mapper.color_to_class_id_map.items():
            r, g, b = rgb
            self.rgb_to_id[r, g, b] = class_id

        # Caricamento percorsi immagini
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                rgb_path = os.path.join(folder_path, 'rgb.jpg')
                labels_path = os.path.join(folder_path, 'labels.png')
                if os.path.exists(rgb_path) and os.path.exists(labels_path):
                    self.samples.append((rgb_path, labels_path, False))
                    if augment:
                        for _ in range(3):
                            self.samples.append((rgb_path, labels_path, True))
                else:
                    print(f"Warning: Missing rgb.jpg or labels.png in {folder_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, labels_path, apply_transform = self.samples[idx]

        # Caricamento immagini
        image = np.array(Image.open(rgb_path).convert("RGB"))
        label_rgb = np.array(Image.open(labels_path).convert("RGB"))

        # Conversione maschera RGB -> class_id (vettorializzata)
        class_id_mask = self.rgb_to_id[
            label_rgb[..., 0],
            label_rgb[..., 1],
            label_rgb[..., 2]
        ]

        # Applica le trasformazioni
        if self.transform and apply_transform:
            augmented = self.transform(image=image, mask=class_id_mask)
            image = augmented['image']
            class_id_mask = augmented['mask']
        elif self.transform:
            val_transform_subset = A.Compose([
                t for t in self.transform.transforms if isinstance(t, (A.Resize, A.Normalize, ToTensorV2))
            ], additional_targets={'mask': 'mask'})
            augmented = val_transform_subset(image=image, mask=class_id_mask)
            image = augmented['image']
            class_id_mask = augmented['mask']
        else:
            image = ToTensorV2()(image=image)['image']
            class_id_mask = torch.from_numpy(class_id_mask).long()

        return image, class_id_mask.long()
