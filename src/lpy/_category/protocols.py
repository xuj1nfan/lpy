"""Structural protocols used by categorical combinators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

A_contra = TypeVar("A_contra", contravariant=True)
B_co = TypeVar("B_co", covariant=True)


class Morphism(Protocol[A_contra, B_co]):
    """Structural type shared by unary callables and :class:`Lambda`."""

    def __call__(self, argument: A_contra, /) -> B_co:
        """Apply the morphism to one value."""
        ...


@runtime_checkable
class SupportsFmap(Protocol):
    """A value belonging to a user-defined functor."""

    def fmap(self, function: Callable[[Any], Any], /) -> Any:
        """Map ``function`` over the value."""
        ...


@runtime_checkable
class SupportsBind(Protocol):
    """A value belonging to a user-defined monad."""

    def bind(self, function: Callable[[Any], Any], /) -> Any:
        """Sequence ``function`` after the value."""
        ...


@runtime_checkable
class SupportsExtract(Protocol):
    """A comonadic value supporting extraction."""

    def extract(self) -> Any:
        """Return the focused value."""
        ...


@runtime_checkable
class SupportsDuplicate(Protocol):
    """A comonadic value supporting duplication."""

    def duplicate(self) -> Any:
        """Return the value in its comonadic context."""
        ...


@runtime_checkable
class SupportsExtend(Protocol):
    """A comonadic value supporting extension."""

    def extend(self, function: Callable[[Any], Any], /) -> Any:
        """Map a context-aware function over the value."""
        ...
