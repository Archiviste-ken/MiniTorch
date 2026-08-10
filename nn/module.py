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

        for value in self.__dict__.values():

            # Direct parameter
            if hasattr(value, "data") and hasattr(value, "grad"):
                params.append(value)

            # Child module
            elif isinstance(value, Module):
                params.extend(value.parameters())

            # List of child modules / parameters
            elif isinstance(value, list):
                for item in value:

                    if hasattr(item, "data") and hasattr(item, "grad"):
                        params.append(item)

                    elif isinstance(item, Module):
                        params.extend(item.parameters())

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