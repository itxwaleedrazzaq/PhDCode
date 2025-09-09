import numpy as np

class ReLULayer:
    def forward(self, x):
        """
        Forward pass for the ReLU layer.
        :param x: Input tensor
        :return: Output tensor after applying ReLU activation
        """
        self.activation = np.maximum(0, x)  # Store activation for backward pass
        return self.activation

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-0.
        :param R: Relevance scores from the next layer
        :return: Relevance scores propagated through active neurons
        """
        # LRP-0: Propagate relevance only through active neurons (activation > 0)
        return R * (self.activation > 0)