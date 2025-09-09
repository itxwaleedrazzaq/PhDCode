import numpy as np

class ReshapeLayer:
    def __init__(self, layer):
        """
        Initialize the Reshape layer.
        :param layer: The Reshape layer (e.g., from TensorFlow or Keras)
        """
        self.layer = layer

    def forward(self, x):
        """
        Forward pass for the Reshape layer.
        :param x: Input tensor (shape: batch_size, ...)
        :return: Reshaped output tensor (shape: batch_size, new_dim1, new_dim2, ...)
        """
        self.original_shape = x.shape  # Store original shape for backward pass
        
        batch_size = self.original_shape[0]
        embed_dim = self.original_shape[-1]
        
        return np.reshape(x, (batch_size, -1, embed_dim))  # Reshape the input

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-0.
        :param R: Relevance scores from the next layer (shape: batch_size, new_dim1, new_dim2, ...)
        :return: Relevance scores reshaped to match the original input shape (shape: batch_size, ...)
        """
        # LRP-0: Simply reshape the relevance to match the original input shape
        return np.reshape(R, self.original_shape)