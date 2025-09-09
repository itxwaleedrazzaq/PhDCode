import numpy as np
import tensorflow as tf

class LambdaLayer:
    def __init__(self, layer, lrp_epsilon=1e-2):
        """
        Initialize the Lambda layer.
        :param layer: The Lambda layer (e.g., from TensorFlow or Keras)
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.layer = layer
        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε

    def forward(self, x):
        """
        Forward pass for the Lambda layer.
        :param x: Input tensor
        :return: Resized output tensor
        """
        self.x = x  # Store input for backward pass
        self.target_shape = (x.shape[1], x.shape[2])  # Dynamically determine target shape
        self.resized_x = tf.image.resize(x, self.target_shape)  # Resize image
        return self.resized_x.numpy()  # Convert to NumPy array

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer
        :return: Relevance scores propagated through the Lambda layer
        """
        # LRP-ε: Add stabilizer term to the denominator
        Z = self.resized_x + self.lrp_epsilon  # Stabilized denominator
        S = R / Z  # Compute sensitivity
        C = S * self.x  # Re-distribute relevance to match original input
        return C.numpy()  # Convert to NumPy array