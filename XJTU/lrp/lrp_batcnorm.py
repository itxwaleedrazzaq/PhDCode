import numpy as np
import tensorflow as tf

class BatchNormalizationLayer:
    def __init__(self, layer, epsilon=1e-5, lrp_epsilon=1e-2):
        """
        Initialize the BatchNormalization layer.
        :param layer: The BatchNormalization layer (e.g., from TensorFlow or Keras)
        :param epsilon: Small constant for numerical stability in batch normalization
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.layer = layer
        weights, beta, mean, variance = layer.get_weights()
        self.gamma = weights  # Scaling factor (γ)
        self.beta = beta  # Shifting factor (β)
        self.mean = mean  # Mean of the batch
        self.variance = variance  # Variance of the batch
        self.epsilon = epsilon  # Small constant for numerical stability
        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε

    def forward(self, x):
        """
        Forward pass for the BatchNormalization layer.
        :param x: Input tensor
        :return: Output tensor after batch normalization
        """
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)  # Store input for backward pass
        out = self.layer.call(self.x)
        return out.numpy()

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer
        :return: Relevance scores propagated through the BatchNormalization layer
        """
        # LRP-ε: Add stabilizer term to the denominator
        Rx = self.gamma * R / np.sqrt(self.variance + self.epsilon + self.lrp_epsilon)
        return Rx