import torch

class ImageProcessor:
    def preprocess(self, image_batch_uint8):
        image_batch_float = image_batch_uint8.float() / 255.0
        return image_batch_float.permute(0, 3, 1, 2)

    def postprocess(self, prediction_tensor_float):
        predicted_classes = torch.argmax(prediction_tensor_float, dim=1)
        predicted_classes = predicted_classes.unsqueeze(-1).to(torch.int64)
        return predicted_classes