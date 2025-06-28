import torch

from ImageProcessor import ImageProcessor
from SegmentationModel import SegmentationModel


class Predictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegmentationModel(num_classes=9).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.processor = ImageProcessor() # Instantiate the processor for internal use

    def predict(self, X):
        # X: (batch_size, rows, cols, 3) uint8 [cite: 15]
        # Must perform all required preprocessing on the batch
        X_tensor = torch.from_numpy(X).to(self.device) # Convert numpy array to tensor
        preprocessed_X = self.processor.preprocess(X_tensor) # Preprocessing

        with torch.no_grad():
            outputs = self.model(preprocessed_X)

        # Must perform all required postprocessing on the results
        predictions = self.processor.postprocess(outputs) # Postprocessing

        # Return value: (batch_size, rows, cols, 1) uint8
        return predictions.cpu().numpy() # Convert back to numpy for return