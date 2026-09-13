"""Shared validation for category-theory combinators."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ..runtime import Lambda, lam
from .protocols import Morphism

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def ensure_unary(
    function: Morphism[A, B],
    name: str,
) -> Lambda[A, B]:
    """Return ``function`` as a unary Lambda or raise a focused error."""
    if isinstance(function, Lambda):
        if function.remaining_arity != 1:
            raise TypeError(
                f"{name} requires a unary Lambda, "
                f"got arity {function.remaining_arity}"
            )
        return function
    if callable(function):
        wrapped = lam(function)
        if wrapped.remaining_arity != 1:
            raise TypeError(
                f"{name} requires a unary callable, "
                f"got arity {wrapped.remaining_arity}"
            )
        return wrapped
    raise TypeError(
        f"{name} expected a callable or Lambda, got {type(function).__name__}"
    )


def require_binary_lambda(
    function: Lambda[A, Lambda[B, C]] | Callable[[A, B], C],
    name: str,
) -> Lambda[A, Lambda[B, C]]:
    """Wrap a fixed binary callable or validate a binary Lambda."""
    if isinstance(function, Lambda):
        wrapped = function
    elif callable(function):
        wrapped = lam(function)
    else:
        raise TypeError(
            f"{name} expected a callable or Lambda, got {type(function).__name__}"
        )
    if wrapped.remaining_arity != 2:
        raise TypeError(
            f"{name} requires a binary Lambda, got arity {wrapped.remaining_arity}"
        )
    return wrapped
