from __future__ import annotations


class Trainer:
    def __init__(self, model: object, optimizer: object, loss_fn: object) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Training loop will be implemented in later lessons.")
