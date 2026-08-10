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
    
    def backward(self):
        """
        Run backpropagation through the computation graph.
        """

        # --------------------------------------------------
        # 1. Build topological ordering
        # --------------------------------------------------

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        # --------------------------------------------------
        # 2. Seed the final node's gradient
        # --------------------------------------------------

        self.grad = 1.0

        # --------------------------------------------------
        # 3. Traverse graph backwards
        # --------------------------------------------------

        for node in reversed(topo):
            node._backward() 

    def __mul__(self, other):
        """
        Multiply two Value objects.

        Builds the forward result and stores the
        local derivative rules needed for backpropagation.
        """

        if not isinstance(other, Value):
            other = Value(other)

        # Forward pass
        out = Value(self.data * other.data)

        # Build graph
        out._prev = {self, other}
        out._op = "*"

        # Backward pass
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __radd__(self, other):
     return self + other


    def __rmul__(self, other):
     return self * other
 
    def __neg__(self):
        out = Value(-self.data)

        out._prev = {self}
        out._op = "neg"

        def _backward():
            self.grad += -out.grad

        out._backward = _backward

        return out
    
    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)