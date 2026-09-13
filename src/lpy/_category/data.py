"""Runtime data representations for categorical structures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

A = TypeVar("A")
B = TypeVar("B")
E = TypeVar("E")
L = TypeVar("L")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")


class Either(Generic[L, R]):
    """Right-biased runtime representation of the coproduct ``L + R``."""

    @property
    def is_left(self) -> bool:
        return isinstance(self, Left)

    @property
    def is_right(self) -> bool:
        return isinstance(self, Right)

    def fmap(self, function: Callable[[R], B]) -> Either[L, B]:
        """Map over the Right variant."""
        if isinstance(self, Right):
            return Right(function(self.value))
        return cast("Either[L, B]", self)

    def map(self, function: Callable[[R], B]) -> Either[L, B]:
        """Compatibility alias for :meth:`fmap`."""
        return self.fmap(function)

    def bind(self, function: Callable[[R], Either[L, B]]) -> Either[L, B]:
        """Sequence a right-biased computation."""
        if isinstance(self, Right):
            result = function(self.value)
            if not isinstance(result, Either):
                raise TypeError("Either.bind function must return an Either")
            return result
        return cast("Either[L, B]", self)


@dataclass(frozen=True, slots=True)
class Left(Either[L, Any]):
    """Left injection of a coproduct."""

    value: L


@dataclass(frozen=True, slots=True)
class Right(Either[Any, R]):
    """Right injection of a coproduct."""

    value: R


class Maybe(Generic[T]):
    """Runtime option type for computations that may not produce a value."""

    @property
    def is_some(self) -> bool:
        return isinstance(self, Some)

    @property
    def is_nothing(self) -> bool:
        return isinstance(self, _NothingType)

    def fmap(self, function: Callable[[T], B]) -> Maybe[B]:
        if isinstance(self, Some):
            return Some(function(self.value))
        return cast("Maybe[B]", self)

    def map(self, function: Callable[[T], B]) -> Maybe[B]:
        """Compatibility alias for :meth:`fmap`."""
        return self.fmap(function)

    def bind(self, function: Callable[[T], Maybe[B]]) -> Maybe[B]:
        if isinstance(self, Some):
            result = function(self.value)
            if not isinstance(result, Maybe):
                raise TypeError("Maybe.bind function must return a Maybe")
            return result
        return cast("Maybe[B]", self)


@dataclass(frozen=True, slots=True)
class Some(Maybe[T]):
    """Present value of :class:`Maybe`."""

    value: T


class _NothingType(Maybe[Any]):
    _instance: _NothingType | None = None

    def __new__(cls) -> _NothingType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Nothing"


Nothing: Maybe[Any] = _NothingType()


@dataclass(frozen=True, slots=True)
class Env(Generic[E, A]):
    """Coreader/environment comonad ``W[A] = (E, A)``."""

    env: E
    value: A

    def extract(self) -> A:
        return self.value

    def duplicate(self) -> Env[E, Env[E, A]]:
        return Env(self.env, self)

    def extend(self, function: Callable[[Env[E, A]], B]) -> Env[E, B]:
        return Env(self.env, function(self))

    def fmap(self, function: Callable[[A], B]) -> Env[E, B]:
        return Env(self.env, function(self.value))

    def map(self, function: Callable[[A], B]) -> Env[E, B]:
        """Compatibility alias for :meth:`fmap`."""
        return self.fmap(function)


class Store(Generic[S, A]):
    """Store comonad ``W[A] = (S -> A, S)``."""

    def __init__(self, pos: S, accessor: Callable[[S], A]) -> None:
        self.pos = pos
        self.accessor = accessor

    def extract(self) -> A:
        return self.accessor(self.pos)

    def duplicate(self) -> Store[S, Store[S, A]]:
        return Store(self.pos, lambda state: Store(state, self.accessor))

    def extend(self, function: Callable[[Store[S, A]], B]) -> Store[S, B]:
        return Store(
            self.pos,
            lambda state: function(Store(state, self.accessor)),
        )

    def fmap(self, function: Callable[[A], B]) -> Store[S, B]:
        return Store(self.pos, lambda state: function(self.accessor(state)))

    def map(self, function: Callable[[A], B]) -> Store[S, B]:
        """Compatibility alias for :meth:`fmap`."""
        return self.fmap(function)


@dataclass(frozen=True, slots=True)
class Fix(Generic[T]):
    """Least fixed point of an endofunctor: ``Fix[F] ~= F[Fix[F]]``."""

    unfix: T
