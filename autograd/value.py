from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Value:
    data: float
    grad: float = 0.0

    def zero_grad(self) -> None:
        self.grad = 0.0
