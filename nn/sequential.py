from __future__ import annotations

from typing import Iterable

from .module import Module


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self.layers = list(modules)

    def parameters(self) -> list[object]:
        parameters: list[object] = []
        for module in self.layers:
            parameters.extend(module.parameters())
        return parameters

    def modules(self) -> Iterable[Module]:
        return list(self.layers)
