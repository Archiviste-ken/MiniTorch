from __future__ import annotations


class LRScheduler:
    def __init__(self, optimizer: object) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        raise NotImplementedError("Learning rate scheduling will be implemented in later lessons.")
