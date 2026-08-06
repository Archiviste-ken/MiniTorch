from __future__ import annotations

from collections.abc import Iterable, Iterator


class DataLoader:
    def __init__(self, dataset: Iterable[object], batch_size: int = 1, shuffle: bool = False) -> None:
        self.dataset = list(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[object]]:
        for index in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[index : index + self.batch_size]
