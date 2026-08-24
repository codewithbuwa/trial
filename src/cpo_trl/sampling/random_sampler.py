import random
from typing import TypeVar

T = TypeVar("T")


def shuffled_copy(items: list[T], *, seed: int) -> list[T]:
    copied = list(items)
    random.Random(seed).shuffle(copied)
    return copied


__all__ = ["shuffled_copy"]
