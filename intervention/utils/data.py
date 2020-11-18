from typing import Generic, TypeVar

from adt import Case, adt

L = TypeVar("L")
R = TypeVar("R")


@adt
class Either(Generic[L, R]):
    LEFT: Case[L]
    RIGHT: Case[R]

    @property
    def is_left(self) -> bool:
        return self.match(left=lambda _: True, right=lambda _: False)

    @property
    def is_right(self) -> bool:
        return not self.is_left
