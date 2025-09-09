import numpy as np
import tensorflow as tf

class Conv1DLayer:
    def __init__(self, layer, gamma=0.5):
        self.layer = layer
        self.weights = tf.convert_to_tensor(layer.get_weights()[0], dtype=tf.float32)
        self.biases = tf.convert_to_tensor(layer.get_weights()[1], dtype=tf.float32)
        self.strides = layer.strides
        self.padding = layer.padding
        self.dilation_rate = layer.dilation_rate
        self.lrp_gamma = gamma  # Gamma term for LRP-γ

    def forward(self, x):
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)  # Store input for backward pass
        self.output = self.layer.call(self.x)
        return self.output.numpy()

    def backward(self, R,gamma=0.3):
        """
        LRP-γ rule for relevance propagation.
        :param R: Relevance scores from the next layer.
        :param gamma: Weighting factor for positive contributions.
        :return: Propagated relevance scores.
        """
        # Separate positive and negative weights
        weights_pos = tf.maximum(0, self.weights)  # Positive weights
        weights_neg = tf.minimum(0, self.weights)  # Negative weights

        # Compute positive and negative contributions
        Z_pos = tf.nn.conv1d(self.x, weights_pos, stride=self.strides, padding=self.padding.upper(), dilations=self.dilation_rate)
        Z_neg = tf.nn.conv1d(self.x, weights_neg, stride=self.strides, padding=self.padding.upper(), dilations=self.dilation_rate)

        # Apply LRP-γ rule: Z = Z_pos + gamma * Z_pos - Z_neg
        Z = Z_pos + gamma * Z_pos - Z_neg

        # Stabilize denominator with a small epsilon
        epsilon = 1e-7
        Z = Z + epsilon

        # Normalize relevance scores
        S = R / Z

        # Propagate relevance using transposed convolution
        input_shape = tf.shape(self.x)
        C = tf.nn.conv1d_transpose(S, self.weights + gamma * weights_pos, input_shape, strides=self.strides, padding=self.padding.upper(), dilations=self.dilation_rate)
        return C.numpy()
