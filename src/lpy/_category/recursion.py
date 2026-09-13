"""Recursion schemes over explicitly functorial pattern values."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from ..runtime import Lambda, lam
from .common import ensure_unary
from .data import Fix
from .functor import functor_for
from .protocols import Morphism

A = TypeVar("A")
B = TypeVar("B")


def cata(
    algebra: Morphism[object, A],
) -> Lambda[Fix[object], A]:
    """Build a catamorphism from an F-algebra.

    Evaluation uses Python recursion and is therefore intended for finite,
    moderately deep structures.
    """
    algebra_lambda: Lambda[object, A] = ensure_unary(algebra, "cata algebra")

    @lam
    def fold(term: Fix[object]) -> A:
        if not isinstance(term, Fix):
            raise TypeError(f"cata expects a Fix, got {type(term).__name__}")
        mapped = functor_for(term.unfix).fmap(
            cast("Lambda[Any, Any]", fold),
            term.unfix,
        )
        return algebra_lambda(mapped)

    return fold


def ana(
    coalgebra: Morphism[A, object],
) -> Lambda[A, Fix[object]]:
    """Build an anamorphism from an F-coalgebra.

    Evaluation uses Python recursion and is therefore intended for finite,
    moderately deep structures.
    """
    coalgebra_lambda: Lambda[A, object] = ensure_unary(
        coalgebra, "ana coalgebra"
    )

    @lam
    def unfold(seed: A) -> Fix[object]:
        layer = coalgebra_lambda(seed)
        mapped = functor_for(layer).fmap(
            cast("Lambda[Any, Any]", unfold),
            layer,
        )
        return Fix(mapped)

    return unfold


@lam
def hylo(
    algebra: Morphism[object, B],
    coalgebra: Morphism[A, object],
) -> Lambda[A, B]:
    """Compose an anamorphism and catamorphism without constructing Fix."""
    algebra_lambda: Lambda[object, B] = ensure_unary(algebra, "hylo algebra")
    coalgebra_lambda: Lambda[A, object] = ensure_unary(
        coalgebra, "hylo coalgebra"
    )

    @lam
    def refold(seed: A) -> B:
        layer = coalgebra_lambda(seed)
        mapped = functor_for(layer).fmap(
            cast("Lambda[Any, Any]", refold),
            layer,
        )
        return algebra_lambda(mapped)

    return refold
