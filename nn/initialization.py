from __future__ import annotations

from autograd.value import Value


def zeros(count: int) -> list[Value]:
    return [Value(0.0) for _ in range(count)]
