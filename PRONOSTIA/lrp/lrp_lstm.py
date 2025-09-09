import numpy as np

class LSTMLayer:
    def __init__(self, layer, lrp_epsilon=1e-2):
        """
        Initialize the LSTM layer.
        :param layer: The LSTM layer (e.g., from TensorFlow or Keras)
        :param lrp_epsilon: Stabilizer term for LRP-ε
        """
        weights = layer.get_weights()
        
        self.W_xh = weights[0]  # Input-to-hidden weights (shape: input_size, 4 * hidden_size)
        self.W_hh = weights[1]  # Hidden-to-hidden weights (shape: hidden_size, 4 * hidden_size)
        self.b = weights[2]     # Bias (shape: 4 * hidden_size)

        # Split weights into input, forget, cell, and output gates
        self.W_i, self.W_f, self.W_c, self.W_o = np.split(self.W_xh, 4, axis=1)
        self.U_i, self.U_f, self.U_c, self.U_o = np.split(self.W_hh, 4, axis=1)

        self.lrp_epsilon = lrp_epsilon  # Stabilizer term for LRP-ε

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for LSTM to store activations for LRP.
        :param x: Input tensor
        :param h_prev: Previous hidden state
        :param c_prev: Previous cell state
        :return: Current hidden state (h) and cell state (c)
        """
        i = self.sigmoid(np.dot(x, self.W_i) + np.dot(h_prev, self.U_i) + self.b[:self.U_i.shape[1]])
        f = self.sigmoid(np.dot(x, self.W_f) + np.dot(h_prev, self.U_f) + self.b[self.U_i.shape[1]:2*self.U_i.shape[1]])
        c_tilde = np.tanh(np.dot(x, self.W_c) + np.dot(h_prev, self.U_c) + self.b[2*self.U_i.shape[1]:3*self.U_i.shape[1]])
        o = self.sigmoid(np.dot(x, self.W_o) + np.dot(h_prev, self.U_o) + self.b[3*self.U_i.shape[1]:])

        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)

        self.last_input = x
        self.last_h_prev = h_prev
        self.last_c_prev = c_prev
        self.last_i, self.last_f, self.last_c, self.last_o = i, f, c, o

        return h, c

    def backward(self, R):
        """
        Backward pass for relevance propagation using LRP-ε.
        :param R: Relevance scores from the next layer
        :return: Relevance scores propagated through the LSTM layer
        """
        # Input validation
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            raise ValueError("Input relevance scores (R) contain NaN or inf values.")

        epsilon = self.lrp_epsilon * np.max(np.abs(self.W_c))  # Relative epsilon

        # Compute relevance distributions for all gates
        weight_abs_i = np.abs(self.W_i)
        weight_abs_f = np.abs(self.W_f)
        weight_abs_c = np.abs(self.W_c)
        weight_abs_o = np.abs(self.W_o)

        relevance_distribution_i = weight_abs_i / (np.sum(weight_abs_i, axis=0, keepdims=True) + epsilon)
        relevance_distribution_f = weight_abs_f / (np.sum(weight_abs_f, axis=0, keepdims=True) + epsilon)
        relevance_distribution_c = weight_abs_c / (np.sum(weight_abs_c, axis=0, keepdims=True) + epsilon)
        relevance_distribution_o = weight_abs_o / (np.sum(weight_abs_o, axis=0, keepdims=True) + epsilon)

        # Combine relevance distributions
        relevance_distribution = (
            relevance_distribution_i + relevance_distribution_f + relevance_distribution_c + relevance_distribution_o
        ) / 4

        # Compute relevance for input
        R_prev = np.dot(R, relevance_distribution.T)
        R_x = R_prev[:, :self.last_input.shape[-1]]  # Input relevance

        return R_x
        
    def sigmoid(self, x):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))