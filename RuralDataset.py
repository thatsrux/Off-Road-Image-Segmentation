import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from LabelMapper import LabelMapper

class RuralDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapper = LabelMapper()
        self.samples = []

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
        image = Image.open(rgb_path).convert("RGB")
        label_image = Image.open(labels_path).convert("RGB")

        # Resize immagini (solo se Resize è presente nella transform)
        resize_size = None
        if self.transform is not None:
            for t in getattr(self.transform, 'transforms', []):
                if t.__class__.__name__ == 'Resize':
                    resize_size = t.size if hasattr(t, 'size') else t.args[0]
                    break
        if resize_size:
            image = image.resize((resize_size[1], resize_size[0]), Image.BILINEAR)
            label_image = label_image.resize((resize_size[1], resize_size[0]), Image.NEAREST)

        # Prepara la mappa delle classi
        label_np = np.array(label_image)
        class_id_mask = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)
        for r in range(label_np.shape[0]):
            for c in range(label_np.shape[1]):
                class_id_mask[r, c] = self.label_mapper.rgb_to_class_id(tuple(label_np[r, c]))

        # Applica trasformazione solo se richiesto
        if self.transform and apply_transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.from_numpy(class_id_mask).long()
        return image, label_tensor

