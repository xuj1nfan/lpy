"""Explicit functor adapters and generic mapping."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from ..runtime import Lambda, lam
from .common import ensure_unary
from .data import Either, Env, Maybe, Store
from .protocols import Morphism, SupportsFmap

A = TypeVar("A")
B = TypeVar("B")

Predicate = Callable[[object], bool]
Mapper = Callable[[Lambda[Any, Any], object], object]


@dataclass(frozen=True, slots=True, eq=False)
class FunctorAdapter:
    """Runtime dictionary describing how to map one functor family."""

    name: str
    _accepts: Predicate = field(repr=False)
    _mapper: Mapper = field(repr=False)

    def accepts(self, value: object) -> bool:
        """Return whether this adapter owns ``value``."""
        return self._accepts(value)

    def fmap(self, function: Lambda[Any, Any], value: object) -> object:
        """Map an already validated unary Lambda over ``value``."""
        if not self.accepts(value):
            raise TypeError(
                f"{self.name} functor does not accept {type(value).__name__}"
            )
        return self._mapper(function, value)


def _method_mapper(function: Lambda[Any, Any], value: object) -> object:
    return cast(SupportsFmap, value).fmap(function)


def _list_mapper(function: Lambda[Any, Any], value: object) -> object:
    return [function(item) for item in cast(list[Any], value)]


def _tuple_mapper(function: Lambda[Any, Any], value: object) -> object:
    return tuple(function(item) for item in cast(tuple[Any, ...], value))


MAYBE_FUNCTOR = FunctorAdapter(
    "Maybe",
    lambda value: isinstance(value, Maybe),
    _method_mapper,
)
EITHER_FUNCTOR = FunctorAdapter(
    "Either",
    lambda value: isinstance(value, Either),
    _method_mapper,
)
ENV_FUNCTOR = FunctorAdapter(
    "Env",
    lambda value: isinstance(value, Env),
    _method_mapper,
)
STORE_FUNCTOR = FunctorAdapter(
    "Store",
    lambda value: isinstance(value, Store),
    _method_mapper,
)
LIST_FUNCTOR = FunctorAdapter(
    "list",
    lambda value: isinstance(value, list),
    _list_mapper,
)
TUPLE_FUNCTOR = FunctorAdapter(
    "tuple",
    lambda value: isinstance(value, tuple),
    _tuple_mapper,
)
PROTOCOL_FUNCTOR = FunctorAdapter(
    "SupportsFmap",
    lambda value: isinstance(value, SupportsFmap),
    _method_mapper,
)

FUNCTOR_ADAPTERS = (
    MAYBE_FUNCTOR,
    EITHER_FUNCTOR,
    ENV_FUNCTOR,
    STORE_FUNCTOR,
    LIST_FUNCTOR,
    TUPLE_FUNCTOR,
    PROTOCOL_FUNCTOR,
)


def functor_for(value: object) -> FunctorAdapter:
    """Resolve the explicit adapter responsible for a runtime value."""
    for adapter in FUNCTOR_ADAPTERS:
        if adapter.accepts(value):
            return adapter
    raise TypeError(f"fmap does not support container of type {type(value).__name__}")


@lam
def fmap(
    function: Morphism[A, B],
    container: object,
) -> object:
    """Lift a morphism through the functor owning ``container``.

    User-defined containers opt in through :class:`SupportsFmap`; callers are
    responsible for ensuring that implementation obeys the functor laws.
    """
    function_lambda: Lambda[A, B] = ensure_unary(function, "fmap function")
    return functor_for(container).fmap(function_lambda, container)
