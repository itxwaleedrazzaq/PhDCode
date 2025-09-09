import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.layers import (
    Dense, Conv2D, Flatten, Activation, MaxPooling2D, BatchNormalization,
    Add, Lambda, Reshape, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Concatenate, MultiHeadAttention, LSTM, LayerNormalization, InputLayer
)
from .lrp_batcnorm import BatchNormalizationLayer
from .lrp_conv2d import Conv2DLayer
from .lrp_actv import ReLULayer
from .lrp_flatten import FlattenLayer
from .lrp_lambda import LambdaLayer
from .lrp_add import AddLayer
from .lrp_maxpool2d import MaxPooling2DLayer
from .lrp_dense import DenseLayer
from .lrp_reshape import ReshapeLayer
from .lrp_conv1d import Conv1DLayer
from .lrp_maxpool1d import MaxPooling1DLayer
from .lrp_globmax1d import GlobalMaxPooling1DLayer
from .lrp_concat import ConcatLayer
from .lrp_mha import MultiheadAttentionLayer
from .lrp_lstm import LSTMLayer
from .lrp_lnorm import LayerNormalizationLayer



class LRP:
    def __init__(self, model):
        super(LRP, self).__init__()
        self.model = model
        self.lrp_layers = self.create_lrp_layers()
        self.layer_outputs = {}  # Dictionary to store outputs of each layer for backward pass

    def create_lrp_layers(self):
        layer_mapping = {
            Conv2D: Conv2DLayer,
            Conv1D: Conv1DLayer,
            MaxPooling2D: MaxPooling2DLayer,
            MaxPooling1D: MaxPooling1DLayer,
            Lambda: LambdaLayer,
            Add: AddLayer,
            Dense: DenseLayer,
            Flatten: FlattenLayer,
            BatchNormalization: BatchNormalizationLayer,
            Activation: lambda _: ReLULayer(),  # No argument needed
            Reshape: ReshapeLayer,
            GlobalMaxPooling1D: lambda _: GlobalMaxPooling1DLayer(),
            Concatenate: lambda _: ConcatLayer(axis=0),
            LayerNormalization: LayerNormalizationLayer,
            MultiHeadAttention: MultiheadAttentionLayer,
            LSTM: LSTMLayer,
        }

        lrp_layers = []
        
        for layer in filter(lambda l: not isinstance(l, InputLayer), self.model.layers):
            layer_class = type(layer)
            if layer_class in layer_mapping:
                lrp_layers.append((layer.name, layer_mapping[layer_class](layer)))
            else:
                raise ValueError(f"Unsupported layer type: {layer_class}")

        return lrp_layers
    

    def find_residual_input_index(self, name):
        # Dictionary lookup for better efficiency
        residual_map = {
            "ires1": "ires1ip",
            "ires2": "ires2ip",
            "tres1": "tres1ip",
            "tres2": "tres2ip",
            "mres1": "mres1ip",
        }
        return next((residual_map[key] for key in residual_map if key in name), None)

    def forward(self, x):
        activations = [x]
        residual_outputs = {}

        for name, layer in self.lrp_layers:
            if isinstance(layer, AddLayer):  # Handle residual connections dynamically
                residual_input_index = self.find_residual_input_index(name)
                residual_input = self.layer_outputs.get(residual_input_index, activations[-1])
                x = layer.forward([activations[-1], residual_input])

                # Store residual output only if the key exists
                if residual_input_index in self.layer_outputs:
                    residual_outputs[name] = x

            elif isinstance(layer, ConcatLayer):  # Handle concatenation layers
                x = layer.forward(*activations[0])  # Tuple unpacking for clarity

            elif isinstance(layer, LSTMLayer):
                batch_size = x.shape[0]
                hidden_size = layer.W_hh.shape[0]

                h_prev = np.zeros((batch_size, hidden_size))
                c_prev = np.zeros_like(h_prev)
                x, _ = layer.forward(activations[-1], h_prev, c_prev)

            else:
                x = layer.forward(activations[-1])

            self.layer_outputs[name] = x  # Store the output for backward pass
            activations.append(x)

        return x, residual_outputs


    def backward(self, relevance, residual_outputs): 
        for name, layer in reversed(self.lrp_layers):
            if isinstance(layer, AddLayer):  # Handle backward propagation for residual blocks
                residual_relevance, main_relevance = layer.backward(relevance)
                
                residual_input_index = self.find_residual_input_index(name)
                if residual_input_index:  # If there's a residual input
                    relevance = main_relevance + residual_relevance + residual_outputs.get(name, 0)
                else:
                    relevance = main_relevance + residual_relevance
            else:
                relevance = layer.backward(relevance)

        return relevance
 

    def explain(self, x, class_index=0):
        output, residual_outputs = self.forward(x)
        R = np.zeros_like(output)
        R[0, class_index] = 1  # Set relevance for the class of interest

        relevance = self.backward(R, residual_outputs)
        return relevance



# Function to plot relevance scores
def plot_relevance(input_data, relevance_scores):



    """
    Plot the relevance scores as a heatmap.
    
    Args:
        input_data: Original input data (e.g., an image).
        relevance_scores: Relevance scores for the input data.
    """
    # Plot the input data
    plt.figure(figsize=(15, 3))
    
    plt.subplot(2, 1, 1)
    plt.imshow(input_data[0, :, :, 0], cmap='gray')
    plt.title("Input Data")
    plt.axis('off')

    
    # Plot the relevance scores
    plt.subplot(2, 1, 2)
    plt.imshow(relevance_scores[0, :, :, 0], cmap='hot', interpolation='nearest')
    plt.title("Relevance Scores")
    plt.axis('off')
    plt.colorbar() # Adds a color legend for better interpretation

    plt.show()

