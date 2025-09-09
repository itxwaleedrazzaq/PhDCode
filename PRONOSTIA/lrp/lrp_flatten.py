import numpy as np

class FlattenLayer:
    def forward(self, x):
        """
        Forward pass for the Flatten layer.
        :param x: Input tensor (e.g., from a Conv2D or MaxPool2D layer)
        :return: Flattened output tensor
        """
        self.input_shape = x.shape  # Store input shape for backward pass
        return x.reshape(x.shape[0], -1)  # Flatten the input (batch_size, -1)

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-0.
        :param R: Relevance scores from the next layer
        :return: Relevance scores reshaped to match the input shape
        """
        # LRP-0: Simply reshape the relevance to match the input shape
        return R.reshape(self.input_shape)  # Reshape relevance to match input shape