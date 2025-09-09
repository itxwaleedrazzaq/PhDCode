import tensorflow as tf
import numpy as np

class GlobalMaxPooling1DLayer:
    def __init__(self, lrp_epsilon=1e-2):
        """
        Initialize the GlobalMaxPooling1D layer.
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.x = None  # Placeholder for input
        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε

    def forward(self, x):
        """
        Forward pass for the GlobalMaxPooling1D layer.
        :param x: Input tensor (shape: batch_size, sequence_length, num_features)
        :return: Output tensor after global max pooling (shape: batch_size, num_features)
        """
        self.x = x  # Store input for backward pass
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        pooled = tf.reduce_max(x_tf, axis=1, keepdims=False)
        return pooled.numpy()

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer (shape: batch_size, num_features)
        :return: Relevance scores propagated through the GlobalMaxPooling1D layer (shape: batch_size, sequence_length, num_features)
        """
        x_tf = tf.convert_to_tensor(self.x, dtype=tf.float32)
        R_tf = tf.convert_to_tensor(R, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            pooled = tf.reduce_max(x_tf, axis=1, keepdims=False)
            loss = tf.reduce_sum(pooled * R_tf)
        grad_tf = tape.gradient(loss, x_tf)
        
        # Add stabilizer term (optional, not strictly necessary for max pooling)
        grad_tf = grad_tf + self.lrp_epsilon
        return grad_tf.numpy()