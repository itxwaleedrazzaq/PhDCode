import numpy as np

class AddLayer:
    def __init__(self, layer):
        """
        Initialize the Add layer.
        :param layer: The layer object (e.g., from TensorFlow or Keras)
        """
        self.layer = layer

    def forward(self, inputs):
        """
        Forward pass for the Add layer.
        :param inputs: List of input tensors to be added
        :return: Sum of the input tensors
        """
        self.inputs = inputs  # Store inputs for backward pass
        return sum(inputs)  # Sum all inputs

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-0.
        :param R: Relevance scores from the next layer
        :return: List of relevance scores distributed equally among inputs
        """
        # LRP-0: Distribute relevance equally among all inputs
        return [R / len(self.inputs) for _ in self.inputs]