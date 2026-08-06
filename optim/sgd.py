from __future__ import annotations


class SGD:
    def __init__(self, parameters: list[object], learning_rate: float = 0.01) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            zero_grad = getattr(parameter, "zero_grad", None)
            if callable(zero_grad):
                zero_grad()

    def step(self) -> None:
        raise NotImplementedError("Optimizer updates will be implemented in later lessons.")
