import numpy as np

class LabelMapper:
    def __init__(self):
        self.color_to_class_id = {
            (255, 255, 255): 0,   # Background
            (1, 88, 255): 1,       # Sky
            (156, 76, 30): 2,     # Rough Trail
            (178, 176, 153): 3,   # Smooth Trail
            (128, 255, 0): 4,     # Traversable grass
            (40, 80, 0): 5,       # High Vegetation
            (0, 160, 0): 6,       # Non Traversable Low Vegetation
            (255, 0, 128): 7,     # Puddle
            (255, 0, 0): 8        # Obstacle
        }
        # Optionally, for debugging/visualization:
        self.class_id_to_color = {v: k for k, v in self.color_to_class_id.items()}

    def rgb_to_class_id(self, rgb_pixel):
        # Converti il pixel in una tupla per la ricerca nel dizionario
        rgb_tuple = tuple(rgb_pixel)
        return self.color_to_class_id.get(rgb_tuple, 0) # Ritorna 0 (Background) se il colore non è mappato

    def class_id_to_rgb(self, class_id):
        return self.class_id_to_color.get(class_id)