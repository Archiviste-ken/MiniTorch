from __future__ import annotations


def accuracy(predictions: list[float], targets: list[float]) -> float:
    if not targets:
        return 0.0
    correct = sum(int(round(pred) == round(target)) for pred, target in zip(predictions, targets))
    return correct / len(targets)
