"""Natural transformations with explicit functor endpoints."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

from ..runtime import Lambda, lam
from .common import ensure_unary
from .data import Either, Left, Maybe, Nothing, Right, Some
from .functor import EITHER_FUNCTOR, LIST_FUNCTOR, MAYBE_FUNCTOR, FunctorAdapter
from .protocols import Morphism

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


class NaturalTransformation(Generic[A, B]):
    """A transformation with explicit source and target functor adapters.

    Endpoint metadata makes vertical composition checkable. Naturality is a
    law supplied by the implementation; :meth:`naturality_holds` can test the
    square for individual morphisms and values but cannot prove it generally.
    """

    def __init__(
        self,
        transform: Morphism[A, B],
        name: str | None = None,
        *,
        source: FunctorAdapter | None = None,
        target: FunctorAdapter | None = None,
    ) -> None:
        inferred_name = getattr(transform, "__name__", "alpha")
        self.name = name or (
            inferred_name if isinstance(inferred_name, str) else "alpha"
        )
        self.transform: Lambda[A, B] = ensure_unary(transform, self.name)
        self.source = source
        self.target = target

    def __call__(self, value: A) -> B:
        return self.transform(value)

    def __matmul__(self, other: object) -> Any:
        """Vertically compose transformations or compose with a Lambda."""
        if isinstance(other, NaturalTransformation):
            if (
                self.source is not None
                and other.target is not None
                and self.source is not other.target
            ):
                raise TypeError(
                    "cannot compose natural transformations: "
                    f"{other.name} targets {other.target.name!r}, but "
                    f"{self.name} expects {self.source.name!r}"
                )
            return NaturalTransformation(
                self.transform @ other.transform,
                name=f"{self.name}_after_{other.name}",
                source=other.source,
                target=self.target,
            )
        if isinstance(other, Lambda):
            return self.transform @ other
        if callable(other):
            return self.transform @ cast("Lambda[Any, Any]", lam(other))
        return NotImplemented

    def __rmatmul__(self, other: object) -> Any:
        """Compose a Lambda or callable after this transformation."""
        if isinstance(other, Lambda):
            return other @ self.transform
        if callable(other):
            return cast("Lambda[Any, Any]", lam(other)) @ self.transform
        return NotImplemented

    def __repr__(self) -> str:
        endpoints = ""
        if self.source is not None or self.target is not None:
            source = None if self.source is None else self.source.name
            target = None if self.target is None else self.target.name
            endpoints = f", source={source!r}, target={target!r}"
        return f"NaturalTransformation({self.name!r}{endpoints})"

    def naturality_holds(self, morphism: Morphism[Any, Any], value: A) -> bool:
        """Check the naturality square for one morphism and input value."""
        if self.source is None or self.target is None:
            raise TypeError("naturality checks require source and target adapters")
        arrow: Lambda[Any, Any] = ensure_unary(morphism, "naturality morphism")
        left = self.target.fmap(arrow, self(value))
        source_value = cast(A, self.source.fmap(arrow, value))
        right = self(source_value)
        return left == right


def _maybe_to_list(maybe: Maybe[A]) -> list[A]:
    if isinstance(maybe, Some):
        return [maybe.value]
    return []


maybe_to_list: NaturalTransformation[Maybe[Any], list[Any]] = NaturalTransformation(
    _maybe_to_list,
    name="maybe_to_list",
    source=MAYBE_FUNCTOR,
    target=LIST_FUNCTOR,
)


def _list_to_maybe(items: list[A]) -> Maybe[A]:
    return Some(items[0]) if items else Nothing


list_to_maybe: NaturalTransformation[list[Any], Maybe[Any]] = NaturalTransformation(
    _list_to_maybe,
    name="list_to_maybe",
    source=LIST_FUNCTOR,
    target=MAYBE_FUNCTOR,
)


def _either_to_maybe(either: Either[Any, A]) -> Maybe[A]:
    if isinstance(either, Right):
        return Some(either.value)
    if isinstance(either, Left):
        return Nothing
    raise TypeError(f"expected an Either, got {type(either).__name__}")


either_to_maybe: NaturalTransformation[
    Either[Any, Any], Maybe[Any]
] = NaturalTransformation(
    _either_to_maybe,
    name="either_to_maybe",
    source=EITHER_FUNCTOR,
    target=MAYBE_FUNCTOR,
)
