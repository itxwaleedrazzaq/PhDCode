import numpy as np
import tensorflow as tf

class DenseLayer:
    def __init__(self, layer, lrp_epsilon=1e-2):
        """
        Initialize the Dense layer.
        :param layer: The Dense layer (e.g., from TensorFlow or Keras)
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.layer = layer
        self.weights, self.biases = layer.get_weights()
        self.activation = layer.activation if hasattr(layer, 'activation') else None
        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε

    def forward(self, x):
        """
        Forward pass through the Dense layer.
        :param x: Input tensor
        """
        self.x = x
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        out = self.layer.call(x)
        return out.numpy()

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer
        """
        # Input validation
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            raise ValueError("Input relevance scores (R) contain NaN or inf values.")

        # Compute pre-activation values before activation
        pre_activation = self.x @ self.weights + self.biases

        # Check if activation is applied (e.g., ReLU, Sigmoid, etc.)
        if self.activation is not None and self.activation != 'linear':
            # Apply ReLU (or other activation) masking (only propagate through active neurons)
            relu_mask = pre_activation > 0  # Neurons that were activated
            R *= relu_mask  # Mask the relevance to propagate only through active neurons

        # Compute Z (denominator for relevance propagation)
        Z = self.x @ self.weights + self.biases

        # Stabilize denominator with a relative epsilon
        epsilon = self.lrp_epsilon * np.max(np.abs(Z))
        Z_safe = Z + epsilon

        # Compute S (sensitivity for relevance propagation)
        S = R / Z_safe  # Shape: (1, 32)

        # Compute C (contributions to previous layer)
        C = S @ self.weights.T  # Shape: (1, 64)

        return self.x * C