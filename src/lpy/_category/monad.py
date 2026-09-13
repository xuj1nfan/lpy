"""Monad operations and Kleisli composition."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from ..core import App, Term, parse
from ..runtime import Lambda, lam
from .common import ensure_unary
from .data import Either, Maybe, Some
from .protocols import Morphism, SupportsBind

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


CHURCH_SOME: Term = parse(r"\x n s. s x")
KLEISLI_COMPOSE_TERM: Term = parse(r"\f g x. bind (f x) g")


@lam(term=CHURCH_SOME)
def pure_maybe(value: A) -> Maybe[A]:
    """Inject a value into the Maybe monad."""
    return Some(value)


def _bind_sequence(
    container: list[Any] | tuple[Any, ...],
    function: Lambda[Any, Any],
) -> list[Any] | tuple[Any, ...]:
    expected = type(container)
    flattened: list[Any] = []
    for item in container:
        result = function(item)
        if not isinstance(result, expected):
            raise TypeError(
                f"{expected.__name__} bind function must return "
                f"a {expected.__name__}"
            )
        flattened.extend(result)
    return expected(flattened)


@lam
def bind(
    container: object,
    function: Morphism[A, object],
) -> object:
    """Sequence a monadic value with a value-producing morphism.

    Maybe, Either, list, and tuple validate their result family. Custom
    values must opt in through :class:`SupportsBind` and enforce their own
    monad-specific result invariant.
    """
    function_lambda: Lambda[Any, Any] = ensure_unary(function, "bind function")
    return _bind_value(container, function_lambda)


def _bind_value(
    container: object,
    function: Lambda[Any, Any],
) -> object:
    if isinstance(container, (Maybe, Either)):
        return container.bind(function)
    if isinstance(container, (list, tuple)):
        return _bind_sequence(container, function)
    if isinstance(container, SupportsBind):
        return container.bind(function)
    raise TypeError(f"bind does not support monad of type {type(container).__name__}")


@lam(term=KLEISLI_COMPOSE_TERM)
def kleisli_compose(
    first: Morphism[A, object],
    second: Morphism[B, object],
) -> Lambda[A, object]:
    """Compose two Kleisli arrows, conventionally written ``>=>``."""
    first_lambda: Lambda[A, object] = ensure_unary(
        first, "kleisli_compose first argument"
    )
    second_lambda: Lambda[B, object] = ensure_unary(
        second, "kleisli_compose second argument"
    )
    symbolic = App(
        App(KLEISLI_COMPOSE_TERM, first_lambda.term),
        second_lambda.term,
    )

    @lam(term=symbolic.normalize())
    def kleisli_arrow(value: A) -> object:
        return _bind_value(
            first_lambda(value),
            cast("Lambda[Any, Any]", second_lambda),
        )

    return kleisli_arrow
