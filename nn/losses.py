from __future__ import annotations

from autograd.value import Value


def mse_loss(predictions: list[Value], targets: list[Value]) -> Value:
    raise NotImplementedError("Loss functions will be implemented in later lessons.")
