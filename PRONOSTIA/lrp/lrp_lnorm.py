import numpy as np

class LayerNormalizationLayer:
    def __init__(self, layer, epsilon=1e-2):
        """
        Initialize the LayerNormalization layer.
        :param weights: Scaling factor (γ)
        :param beta: Shifting factor (β)
        :param epsilon: Small constant for numerical stability in layer normalization
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.gamma, self.beta = layer.get_weights()
        self.epsilon = epsilon

    def forward(self, x):
        """
        Forward pass for the LayerNormalization layer.
        :param x: Input tensor
        :return: Output tensor after layer normalization
        """
        self.x = x  # Store input for backward pass
        self.mean = np.mean(x, axis=-1, keepdims=True)  # Mean of the input
        self.variance = np.var(x, axis=-1, keepdims=True)  # Variance of the input
        self.x_normalized = (x - self.mean) / np.sqrt(self.variance + self.epsilon)  # Normalized input
        return self.gamma * self.x_normalized + self.beta  # Scale and shift

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer
        :return: Relevance scores propagated through the LayerNormalization layer
        """
        # LRP-ε: Add stabilizer term to the denominator
        Rx = self.gamma * R / np.sqrt(self.variance + self.epsilon)
        return Rx