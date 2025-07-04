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
        self.samples = []
        self.augment = augment

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                rgb_path = os.path.join(folder_path, 'rgb.jpg')
                labels_path = os.path.join(folder_path, 'labels.png')
                if os.path.exists(rgb_path) and os.path.exists(labels_path):
                    # Aggiungi versione originale
                    self.samples.append((rgb_path, labels_path, False))
                    # Aggiungi versione aumentata se richiesto
                    if augment:
                        self.samples.append((rgb_path, labels_path, True))
                        self.samples.append((rgb_path, labels_path, True))
                        self.samples.append((rgb_path, labels_path, True))
                else:
                    print(f"Warning: Missing rgb.jpg or labels.png in {folder_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, labels_path, apply_transform = self.samples[idx]
        image = np.array(Image.open(rgb_path).convert("RGB"))
        label_image = np.array(Image.open(labels_path).convert("RGB"))

        # Prepara la mappa delle classi
        class_id_mask = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
        for r in range(label_image.shape[0]):
            for c in range(label_image.shape[1]):
                class_id_mask[r, c] = self.label_mapper.rgb_to_class_id(tuple(label_image[r, c]))


        # Applica trasformazione solo se richiesto e disponibile
        if self.transform and apply_transform:
            augmented = self.transform(image=image, mask=class_id_mask)
            image = augmented['image']
            class_id_mask = augmented['mask']
        elif self.transform and not apply_transform:
             # Applica solo le trasformazioni di validazione (Resize, Normalize, ToTensor)
             val_transform_subset = A.Compose([
                t for t in self.transform.transforms if isinstance(t, (A.Resize, A.Normalize, ToTensorV2))
             ], additional_targets={'mask': 'mask'})
             augmented = val_transform_subset(image=image, mask=class_id_mask)
             image = augmented['image']
             class_id_mask = augmented['mask']
        else:
             # Fallback if no transform is provided (should not happen with the current usage)
             image = ToTensorV2()(image=image)['image']
             class_id_mask = torch.from_numpy(class_id_mask).long()


        label_tensor = class_id_mask.long()
        return image, label_tensor