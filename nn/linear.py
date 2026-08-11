"""
========================================================

Lesson Connection
-----------------
Lesson 2: Neurons and Linear Layers

PyTorch Equivalent:
torch.nn.Linear

Purpose:
Implement a fully-connected linear layer.

Formula:
y = Wx + b

========================================================
"""

import random

from autograd.value import Value
from nn.module import Module


class Linear(Module):
    """
    Fully-connected linear layer.

    Maps:
        in_features → out_features

    Example:
        Linear(3, 4)

    means:
        3 inputs
        4 neurons
        4 outputs
    """

    def __init__(self, in_features, out_features):

        self.in_features = in_features
        self.out_features = out_features

        # One weight vector for every neuron.
        self.weights = [
            [
                Value(random.uniform(-1, 1))
                for _ in range(in_features)
            ]
            for _ in range(out_features)
        ]

        # One bias for every neuron.
        self.biases = [
            Value(random.uniform(-1, 1))
            for _ in range(out_features)
        ]

    def forward(self, inputs):

        outputs = []

        for weights, bias in zip(self.weights, self.biases):

            total = bias

            for weight, input_value in zip(weights, inputs):
                total += weight * input_value

            outputs.append(total)

        return outputs