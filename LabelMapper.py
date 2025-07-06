import numpy as np
from PIL import Image
import torch

class LabelMapper:
    def __init__(self):
        self.color_to_class_id_map = { # Renamed to avoid confusion
            (255, 255, 255): 0,   # Background
            (1, 88, 255): 1,      # Sky
            (156, 76, 30): 2,     # Rough Trail
            (178, 176, 153): 3,   # Smooth Trail
            (128, 255, 0): 4,     # Traversable grass
            (40, 80, 0): 5,       # High Vegetation
            (0, 160, 0): 6,       # Non Traversable Low Vegetation
            (255, 0, 128): 7,     # Puddle
            (255, 0, 0): 8        # Obstacle
        }
        # Optionally, for debugging/visualization:
        self.class_id_to_color = {v: k for k, v in self.color_to_class_id_map.items()}

    def rgb_to_class_id(self, rgb_pixel):
        # Converti il pixel in una tupla per la ricerca nel dizionario
        rgb_tuple = tuple(rgb_pixel)
        return self.color_to_class_id_map.get(rgb_tuple, 0) # Ritorna 0 (Background) se il colore non è mappato

    def class_id_to_rgb(self, class_id):
        return self.class_id_to_color.get(class_id)

    def color_to_class_id(self, labels_image: Image.Image) -> torch.Tensor:
        """
        Converts a PIL RGB image of labels to a PyTorch tensor of class IDs.
        Pixels with colors not defined in color_to_class_id_map will be mapped to 0 (Background).
        """
        labels_array = np.array(labels_image)
        height, width, _ = labels_array.shape
        class_id_map = np.zeros((height, width), dtype=np.int64)

        # Iterate through the color map and assign class IDs
        for color_tuple, class_id in self.color_to_class_id_map.items():
            # Create a boolean mask where the pixels match the current color
            match = np.all(labels_array == np.array(color_tuple).reshape(1, 1, 3), axis=2)
            class_id_map[match] = class_id

        return torch.from_numpy(class_id_map)

    def class_id_to_rgb_image(self, class_id_array: np.ndarray) -> np.ndarray:
        """
        Converts a NumPy array of class IDs to an RGB image array.
        """
        height, width = class_id_array.shape
        rgb_image_array = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color_tuple in self.class_id_to_color.items():
            match = (class_id_array == class_id)
            rgb_image_array[match] = np.array(color_tuple)
        return rgb_image_array