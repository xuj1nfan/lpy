"""Composable runtime lenses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from ..runtime import Lambda, lam
from .common import ensure_unary
from .protocols import Morphism

A = TypeVar("A")
B = TypeVar("B")
S = TypeVar("S")


class Lens(Generic[S, A]):
    """A getter/setter pair expected to satisfy the three lens laws."""

    def __init__(
        self,
        getter: Morphism[S, A],
        setter: Callable[[S, A], S],
    ) -> None:
        self._get: Lambda[S, A] = ensure_unary(getter, "Lens getter")

        @lam
        def curried_setter(source: S, focus: A) -> S:
            return setter(source, focus)

        self._set: Lambda[S, Lambda[A, S]] = curried_setter

    @property
    def get(self) -> Lambda[S, A]:
        return self._get

    @property
    def set(self) -> Lambda[S, Lambda[A, S]]:
        return self._set

    def modify(
        self,
        function: Morphism[A, A],
    ) -> Lambda[S, S]:
        """Modify the focus while preserving the surrounding structure."""
        function_lambda: Lambda[A, A] = ensure_unary(
            function, "Lens modify function"
        )

        @lam
        def modifier(source: S) -> S:
            return self._set(source)(function_lambda(self._get(source)))

        return modifier

    def __matmul__(self: Lens[A, B], other: Lens[S, A]) -> Lens[S, B]:
        """Compose lenses; ``inner @ outer`` focuses through ``outer``."""
        if not isinstance(other, Lens):
            return NotImplemented

        def composed_getter(source: S) -> B:
            return self.get(other.get(source))

        def composed_setter(source: S, focus: B) -> S:
            inner = self.set(other.get(source))(focus)
            return other.set(source)(inner)

        return Lens(composed_getter, composed_setter)
