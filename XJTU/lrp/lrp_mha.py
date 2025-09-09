import tensorflow as tf
import numpy as np

class MultiheadAttentionLayer:
    def __init__(self, layer):
        self.layer = layer
        self.attention_weights = None
        self.x = None

    def forward(self, x):
        """
        Forward pass to get attention weights and outputs.
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)  # Add batch dimension
        out, self.attention_weights = self.layer.call(x, x, x, return_attention_scores=True)
        self.attention_weights = tf.reduce_mean(self.attention_weights, axis=1)  # Shape: [batch_size, seq_len, seq_len]
        return out.numpy()  # Or any other output needed for forward pass

    def backward(self, relevance):
        """
        Backward pass to redistribute relevance based on attention weights using LRP-0.
        """
        # Input validation
        if tf.reduce_any(tf.math.is_nan(relevance)) or tf.reduce_any(tf.math.is_inf(relevance)):
            raise ValueError("Input relevance scores contain NaN or inf values.")

        # Stabilize attention weights with a small epsilon
        epsilon = 1e-7 * tf.reduce_max(tf.abs(self.attention_weights))
        stabilized_attention_weights = self.attention_weights + epsilon

        # Normalize attention weights to ensure they sum to 1
        normalized_attention_weights = stabilized_attention_weights / tf.reduce_sum(stabilized_attention_weights, axis=-1, keepdims=True)

        # Redistribute relevance based on normalized attention weights
        relevance_input = tf.matmul(normalized_attention_weights, relevance)

        return relevance_input.numpy()