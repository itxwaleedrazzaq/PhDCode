import numpy as np

class ConcatLayer:
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *inputs):
        self.inputs = inputs
        return np.concatenate(inputs, axis=self.axis)

    def backward(self, R):
        R = R.squeeze()
        slices = np.split(R, np.cumsum([inp.shape[self.axis] for inp in self.inputs[:-1]]), axis=self.axis)
        return slices
