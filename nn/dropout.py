from __future__ import annotations

from dataclasses import dataclass

from .module import Module


@dataclass
class Dropout(Module):
    probability: float = 0.5
