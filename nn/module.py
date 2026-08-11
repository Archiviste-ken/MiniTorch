"""
========================================================

Lesson Connection
-----------------
Lessons 1–9: Deep Learning Framework Architecture

PyTorch Equivalent:
torch.nn.Module

Purpose:
Base class for neural-network components.

Provides:
- parameter discovery
- gradient resetting
- callable model interface

========================================================
"""


class Module:
    """
    Base class for all neural-network components.

    A Module can:
    - contain trainable parameters
    - contain child Modules
    - perform a forward computation
    """

    def parameters(self):
        """
        Recursively collect all trainable parameters.
        """

        params = []

        def collect(value):

            # Direct parameter
            if hasattr(value, "data") and hasattr(value, "grad"):
                params.append(value)

            # Child Module
            elif isinstance(value, Module):
                params.extend(value.parameters())

            # Container
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        for value in self.__dict__.values():
            collect(value)

        return params

    def zero_grad(self):
        """
        Reset gradients of all trainable parameters.
        """

        for parameter in self.parameters():
            parameter.grad = 0.0

    def __call__(self, *args, **kwargs):
        """
        Make the Module callable.

        model(x)

        becomes:

        model.forward(x)
        """

        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Define the computation performed by the module.

        Child classes must implement this.
        """

        raise NotImplementedError(
            "Every Module must implement forward()."
        )