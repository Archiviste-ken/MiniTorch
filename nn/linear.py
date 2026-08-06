from __future__ import annotations

from dataclasses import dataclass

from autograd.value import Value

from .module import Module


@dataclass
class Linear(Module):
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        self.weights = [Value(0.0) for _ in range(self.in_features * self.out_features)]
        self.bias = [Value(0.0) for _ in range(self.out_features)]

    def parameters(self) -> list[Value]:
        return [*self.weights, *self.bias]
