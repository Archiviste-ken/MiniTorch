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
        
    def __add__(self, other):
        
        """
        Add two Value objects.

        This performs two jobs:

        1. Compute the numerical result.
        2. Extend the computational graph.
        """

        # Allow expressions like:
        # Value + 3
        if not isinstance(other, Value):
            other = Value(other)

        # Create the output node.
        out = Value(self.data + other.data)

        # Remember how this node was created.
        out._prev = {self, other}
        out._op = "+"

        # Backpropagation rule.
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out   