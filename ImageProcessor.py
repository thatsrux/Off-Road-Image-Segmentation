class ImageProcessor:
    def preprocess(self, image_batch_uint8):
        # image_batch_uint8 shape: (batch_size, rows, cols, 3) uint8 [cite: 15]
        # Convert to float and normalize for model input
        image_batch_float = image_batch_uint8.float() / 255.0
        # Permute to (batch_size, 3, rows, cols) for PyTorch convention if needed
        return image_batch_float.permute(0, 3, 1, 2)

    def postprocess(self, prediction_tensor_float):
        # prediction_tensor_float shape: (batch_size, 9, rows, cols) (logits/probabilities)
        # Get the class with max probability for each pixel
        # Output shape: (batch_size, rows, cols, 1) uint8
        predicted_classes = torch.argmax(prediction_tensor_float, dim=1) # (batch_size, rows, cols)
        predicted_classes = predicted_classes.unsqueeze(-1).byte() # Add channel dim and convert to uint8
        return predicted_classes