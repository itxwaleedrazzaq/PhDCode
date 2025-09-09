import tensorflow as tf
import numpy as np

class MaxPooling2DLayer:
    def __init__(self,layer, lrp_epsilon=1e-2):
        """
        Initialize the MaxPooling2D layer.
        :param pool_size: Size of the pooling window (height, width)
        :param strides: Strides of the pooling operation (height, width)
        :param padding: Padding mode ('SAME' or 'VALID')
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        self.layer = layer
        self.pool_size = layer.pool_size
        self.strides = layer.strides
        self.padding = layer.padding.upper()
        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε

    def forward(self, x):
        """
        Forward pass for the MaxPooling2D layer.
        :param x: Input tensor (shape: batch_size, height, width, channels)
        :return: Output tensor after max pooling (shape: batch_size, pooled_height, pooled_width, channels)
        """
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        pooled = tf.nn.max_pool2d(input=self.x,
                                  ksize=self.pool_size, 
                                  strides=self.strides, 
                                  padding=self.padding)
        return pooled.numpy()

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer (shape: batch_size, pooled_height, pooled_width, channels)
        :return: Relevance scores propagated through the MaxPooling2D layer (shape: batch_size, height, width, channels)
        """
        R = tf.convert_to_tensor(R, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(self.x)
            pooled = tf.nn.max_pool2d(input=self.x,
                                      ksize=self.pool_size,
                                      strides=self.strides,
                                      padding=self.padding)
            loss = tf.reduce_sum(pooled * R)
        grad_tf = tape.gradient(loss, self.x)
        
        # Add stabilizer term (optional, not strictly necessary for max pooling)
        grad_tf = grad_tf + self.lrp_epsilon
        return grad_tf.numpy()