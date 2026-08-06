from __future__ import annotations

from dataclasses import dataclass

from .module import Module


@dataclass
class BatchNorm1d(Module):
    features: int
