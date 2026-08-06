from __future__ import annotations

from dataclasses import dataclass

from .module import Module


@dataclass
class ReLU(Module):
    pass


@dataclass
class Sigmoid(Module):
    pass


@dataclass
class Tanh(Module):
    pass
