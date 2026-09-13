"""Public facade for executable category-theory abstractions.

Python values provide the runtime representation. Selected constants expose
the corresponding non-executing symbolic terms; generic operations that span
multiple runtime families deliberately remain extended or opaque.
"""

from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar, cast, overload

from ._category import comonad as _comonad
from ._category import functor as _functor
from ._category import monad as _monad
from ._category import morphisms as _morphisms
from ._category import recursion as _recursion
from ._category.data import Either, Env, Fix, Left, Maybe, Nothing, Right, Some, Store
from ._category.functor import (
    EITHER_FUNCTOR,
    ENV_FUNCTOR,
    FUNCTOR_ADAPTERS,
    LIST_FUNCTOR,
    MAYBE_FUNCTOR,
    PROTOCOL_FUNCTOR,
    STORE_FUNCTOR,
    TUPLE_FUNCTOR,
    FunctorAdapter,
    functor_for,
)
from ._category.lens import Lens
from ._category.monad import (
    CHURCH_SOME,
    KLEISLI_COMPOSE_TERM,
)
from ._category.morphisms import (
    CHURCH_BIMAP_COPRODUCT,
    CHURCH_BIMAP_PRODUCT,
    CHURCH_CURRY,
    CHURCH_EVAL,
    CHURCH_FANIN,
    CHURCH_FANOUT,
    CHURCH_FST,
    CHURCH_IDENTITY,
    CHURCH_INL,
    CHURCH_INR,
    CHURCH_PAIR,
    CHURCH_SND,
    CHURCH_UNCURRY,
)
from ._category.natural import (
    NaturalTransformation,
    either_to_maybe,
    list_to_maybe,
    maybe_to_list,
)
from ._category.protocols import (
    Morphism,
    SupportsBind,
    SupportsDuplicate,
    SupportsExtend,
    SupportsExtract,
    SupportsFmap,
)
from ._category.recursion import ana, cata
from .core import Term
from .runtime import Lambda, SymbolicKind

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
E = TypeVar("E")
L = TypeVar("L")
S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class _SymbolicOperator(Protocol):
    @property
    def term(self) -> Term: ...

    @property
    def remaining_arity(self) -> int: ...

    @property
    def symbolic_kind(self) -> SymbolicKind: ...

    @property
    def symbolic_reason(self) -> str | None: ...


class _IdentityOperator(_SymbolicOperator, Protocol):
    def __call__(self, value: T, /) -> T: ...

    def __matmul__(self, other: Lambda[T, U], /) -> Lambda[T, U]: ...


class _FstOperator(_SymbolicOperator, Protocol):
    def __call__(self, pair: tuple[T, U], /) -> T: ...


class _SndOperator(_SymbolicOperator, Protocol):
    def __call__(self, pair: tuple[T, U], /) -> U: ...


class _FanoutSecond(Protocol, Generic[T, U_co]):
    def __call__(
        self,
        second: Morphism[T, V],
        /,
    ) -> Lambda[T, tuple[U_co, V]]: ...


class _FanoutOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        first: Morphism[T, U],
        /,
    ) -> _FanoutSecond[T, U]: ...


class _BimapProductSecond(Protocol, Generic[T_contra, U_co]):
    def __call__(
        self,
        second: Morphism[V, W],
        /,
    ) -> Lambda[tuple[T_contra, V], tuple[U_co, W]]: ...


class _BimapProductOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        first: Morphism[T, U],
        /,
    ) -> _BimapProductSecond[T, U]: ...


class _CurryOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        function: Morphism[tuple[T, U], V],
        /,
    ) -> Lambda[T, Lambda[U, V]]: ...


class _UncurryOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        function: Lambda[T, Lambda[U, V]] | Callable[[T, U], V],
        /,
    ) -> Lambda[tuple[T, U], V]: ...


class _EvalOperator(_SymbolicOperator, Protocol):
    def __call__(self, value: tuple[Morphism[T, U], T], /) -> U: ...


class _InlOperator(_SymbolicOperator, Protocol):
    def __call__(self, value: T, /) -> Either[T, Any]: ...


class _InrOperator(_SymbolicOperator, Protocol):
    def __call__(self, value: U, /) -> Either[Any, U]: ...


class _FaninSecond(Protocol, Generic[T, V]):
    def __call__(
        self,
        right: Morphism[U, V],
        /,
    ) -> Lambda[Either[T, U], V]: ...


class _FaninOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        left: Morphism[T, V],
        /,
    ) -> _FaninSecond[T, V]: ...


class _BimapCoproductSecond(Protocol, Generic[T, U]):
    def __call__(
        self,
        right: Morphism[V, W],
        /,
    ) -> Lambda[Either[T, V], Either[U, W]]: ...


class _BimapCoproductOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        left: Morphism[T, U],
        /,
    ) -> _BimapCoproductSecond[T, U]: ...


class _FmapSecond(Protocol, Generic[T, U]):
    @overload
    def __call__(self, container: Maybe[T], /) -> Maybe[U]: ...

    @overload
    def __call__(self, container: Either[L, T], /) -> Either[L, U]: ...

    @overload
    def __call__(self, container: Env[E, T], /) -> Env[E, U]: ...

    @overload
    def __call__(self, container: Store[S, T], /) -> Store[S, U]: ...

    @overload
    def __call__(self, container: list[T], /) -> list[U]: ...

    @overload
    def __call__(self, container: tuple[T, ...], /) -> tuple[U, ...]: ...

    @overload
    def __call__(self, container: SupportsFmap, /) -> object: ...


class _FmapOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        function: Morphism[T, U],
        /,
    ) -> _FmapSecond[T, U]: ...


class _MaybeBind(Protocol, Generic[T_co]):
    def __call__(self, function: Morphism[T_co, Maybe[U]], /) -> Maybe[U]: ...


class _EitherBind(Protocol, Generic[L, T_co]):
    def __call__(
        self,
        function: Morphism[T_co, Either[L, U]],
        /,
    ) -> Either[L, U]: ...


class _ListBind(Protocol, Generic[T_co]):
    def __call__(self, function: Morphism[T_co, list[U]], /) -> list[U]: ...


class _TupleBind(Protocol, Generic[T_co]):
    def __call__(
        self,
        function: Morphism[T_co, tuple[U, ...]],
        /,
    ) -> tuple[U, ...]: ...


class _CustomBind(Protocol):
    def __call__(self, function: Morphism[Any, Any], /) -> object: ...


class _BindOperator(_SymbolicOperator, Protocol):
    @overload
    def __call__(self, container: Maybe[T], /) -> _MaybeBind[T]: ...

    @overload
    def __call__(self, container: Either[L, T], /) -> _EitherBind[L, T]: ...

    @overload
    def __call__(self, container: list[T], /) -> _ListBind[T]: ...

    @overload
    def __call__(self, container: tuple[T, ...], /) -> _TupleBind[T]: ...

    @overload
    def __call__(self, container: SupportsBind, /) -> _CustomBind: ...


class _PureMaybeOperator(_SymbolicOperator, Protocol):
    def __call__(self, value: T, /) -> Maybe[T]: ...


class _KleisliSecond(Protocol, Generic[T_contra]):
    def __call__(
        self,
        second: Morphism[Any, object],
        /,
    ) -> Lambda[T_contra, object]: ...


class _KleisliOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        first: Morphism[T, object],
        /,
    ) -> _KleisliSecond[T]: ...


class _ExtractOperator(_SymbolicOperator, Protocol):
    @overload
    def __call__(self, comonad: Env[E, T], /) -> T: ...

    @overload
    def __call__(self, comonad: Store[S, T], /) -> T: ...

    @overload
    def __call__(self, comonad: SupportsExtract, /) -> object: ...


class _DuplicateOperator(_SymbolicOperator, Protocol):
    @overload
    def __call__(self, comonad: Env[E, T], /) -> Env[E, Env[E, T]]: ...

    @overload
    def __call__(self, comonad: Store[S, T], /) -> Store[S, Store[S, T]]: ...

    @overload
    def __call__(self, comonad: SupportsDuplicate, /) -> object: ...


class _ExtendSecond(Protocol, Generic[U]):
    @overload
    def __call__(self, comonad: Env[E, T], /) -> Env[E, U]: ...

    @overload
    def __call__(self, comonad: Store[S, T], /) -> Store[S, U]: ...

    @overload
    def __call__(self, comonad: SupportsExtend, /) -> object: ...


class _ExtendOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        function: Morphism[Any, U],
        /,
    ) -> _ExtendSecond[U]: ...


class _HyloSecond(Protocol, Generic[U_co]):
    def __call__(
        self,
        coalgebra: Morphism[T, object],
        /,
    ) -> Lambda[T, U_co]: ...


class _HyloOperator(_SymbolicOperator, Protocol):
    def __call__(
        self,
        algebra: Morphism[object, U],
        /,
    ) -> _HyloSecond[U]: ...


identity: _IdentityOperator = cast(_IdentityOperator, _morphisms.identity)
fst: _FstOperator = cast(_FstOperator, _morphisms.fst)
snd: _SndOperator = cast(_SndOperator, _morphisms.snd)
fanout: _FanoutOperator = cast(_FanoutOperator, _morphisms.fanout)
bimap_product: _BimapProductOperator = cast(
    _BimapProductOperator,
    _morphisms.bimap_product,
)
curry: _CurryOperator = cast(_CurryOperator, _morphisms.curry)
uncurry: _UncurryOperator = cast(_UncurryOperator, _morphisms.uncurry)
eval_morphism: _EvalOperator = cast(_EvalOperator, _morphisms.eval_morphism)
inl: _InlOperator = cast(_InlOperator, _morphisms.inl)
inr: _InrOperator = cast(_InrOperator, _morphisms.inr)
fanin: _FaninOperator = cast(_FaninOperator, _morphisms.fanin)
bimap_coproduct: _BimapCoproductOperator = cast(
    _BimapCoproductOperator,
    _morphisms.bimap_coproduct,
)
fmap: _FmapOperator = cast(_FmapOperator, _functor.fmap)
bind: _BindOperator = cast(_BindOperator, _monad.bind)
pure_maybe: _PureMaybeOperator = cast(_PureMaybeOperator, _monad.pure_maybe)
kleisli_compose: _KleisliOperator = cast(
    _KleisliOperator,
    _monad.kleisli_compose,
)
extract: _ExtractOperator = cast(_ExtractOperator, _comonad.extract)
duplicate: _DuplicateOperator = cast(_DuplicateOperator, _comonad.duplicate)
extend: _ExtendOperator = cast(_ExtendOperator, _comonad.extend)
hylo: _HyloOperator = cast(_HyloOperator, _recursion.hylo)

__all__ = [
    "CHURCH_BIMAP_COPRODUCT",
    "CHURCH_BIMAP_PRODUCT",
    "CHURCH_CURRY",
    "CHURCH_EVAL",
    "CHURCH_FANIN",
    "CHURCH_FANOUT",
    "CHURCH_FST",
    "CHURCH_IDENTITY",
    "CHURCH_INL",
    "CHURCH_INR",
    "CHURCH_PAIR",
    "CHURCH_SND",
    "CHURCH_SOME",
    "CHURCH_UNCURRY",
    "EITHER_FUNCTOR",
    "ENV_FUNCTOR",
    "FUNCTOR_ADAPTERS",
    "KLEISLI_COMPOSE_TERM",
    "LIST_FUNCTOR",
    "MAYBE_FUNCTOR",
    "PROTOCOL_FUNCTOR",
    "STORE_FUNCTOR",
    "TUPLE_FUNCTOR",
    "Either",
    "Env",
    "Fix",
    "FunctorAdapter",
    "Left",
    "Lens",
    "Maybe",
    "Morphism",
    "NaturalTransformation",
    "Nothing",
    "Right",
    "Some",
    "Store",
    "SupportsBind",
    "SupportsDuplicate",
    "SupportsExtend",
    "SupportsExtract",
    "SupportsFmap",
    "ana",
    "bimap_coproduct",
    "bimap_product",
    "bind",
    "cata",
    "curry",
    "duplicate",
    "either_to_maybe",
    "eval_morphism",
    "extend",
    "extract",
    "fanin",
    "fanout",
    "fmap",
    "fst",
    "functor_for",
    "hylo",
    "identity",
    "inl",
    "inr",
    "kleisli_compose",
    "list_to_maybe",
    "maybe_to_list",
    "pure_maybe",
    "snd",
    "uncurry",
]
