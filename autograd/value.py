"""
========================================================

Lesson Connection
-----------------
Lesson 3: Backpropagation

PyTorch Equivalent:
torch.Tensor (autograd-enabled)

Purpose:
Represents a single node in the computational graph.
Stores its value, gradient, graph connections,
and the information needed for backpropagation.

========================================================
"""


class Value:
    """
    A single scalar value that participates in automatic differentiation.

    Think of it as a 'smart number'.

    Unlike a normal Python float, a Value remembers:
    - its numerical value
    - where it came from
    - how it was created
    - how to backpropagate gradients
    """

    def __init__(self, data):
        """
        Create a Value node.

        Parameters
        ----------
        data : float
            The numerical value stored in this node.
        """

        # The actual scalar value.
        self.data = float(data)

        # Gradient accumulated during backpropagation.
        self.grad = 0.0

        # Parent nodes that produced this Value.
        self._prev = set()

        # Operation that created this node.
        self._op = ""

        # Function used during backpropagation.
        self._backward = lambda: None