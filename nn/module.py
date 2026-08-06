from __future__ import annotations

from typing import Iterable


class Module:
    def parameters(self) -> list[object]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            zero_grad = getattr(parameter, "zero_grad", None)
            if callable(zero_grad):
                zero_grad()

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def modules(self) -> Iterable[Module]:
        return []
