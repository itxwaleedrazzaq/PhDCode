import tensorflow as tf
import numpy as np

class MaxPooling1DLayer:
    def __init__(self, layer, lrp_epsilon=1e-2):
        """
        Initialize the MaxPooling1D layer.
        :param pool_size: Size of the pooling window
        :param strides: Strides of the pooling operation
        :param padding: Padding mode ('SAME' or 'VALID')
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.layer = layer
        self.pool_size = layer.pool_size[0]
        self.strides = layer.strides[0]
        self.padding = layer.padding.upper()
        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε
        self.x = None  # Placeholder for input

    def forward(self, x):
        """
        Forward pass for the MaxPooling1D layer.
        :param x: Input tensor (shape: batch_size, sequence_length, num_features)
        :return: Output tensor after max pooling (shape: batch_size, pooled_sequence_length, num_features)
        """
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)  # Store input for backward pass
        pooled = tf.nn.pool(input=self.x,
                            window_shape=[self.pool_size], 
                            pooling_type='MAX', 
                            strides=[self.strides], 
                            padding=self.padding)
        return pooled.numpy()

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer (shape: batch_size, pooled_sequence_length, num_features)
        :return: Relevance scores propagated through the MaxPooling1D layer (shape: batch_size, sequence_length, num_features)
        """
        R = tf.convert_to_tensor(R, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(self.x)
            pooled = tf.nn.pool(input=self.x,
                                window_shape=[self.pool_size],
                                pooling_type='MAX',
                                strides=[self.strides],
                                padding=self.padding)
            loss = tf.reduce_sum(pooled * R)
        grad_tf = tape.gradient(loss, self.x)
        
        # Add stabilizer term (optional, not strictly necessary for max pooling)
        grad_tf = grad_tf + self.lrp_epsilon
        return grad_tf.numpy()