"""Products, coproducts, exponentials, and their symbolic encodings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ..core import App, Term, parse
from ..runtime import Lambda, lam
from .common import ensure_unary, require_binary_lambda
from .data import Either, Left, Right
from .protocols import Morphism

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")


CHURCH_IDENTITY: Term = parse(r"\x. x")
CHURCH_PAIR: Term = parse(r"\x y p. p x y")
CHURCH_FST: Term = parse(r"\p. p (\x y. x)")
CHURCH_SND: Term = parse(r"\p. p (\x y. y)")
CHURCH_FANOUT: Term = parse(r"\f g x p. p (f x) (g x)")
CHURCH_BIMAP_PRODUCT: Term = parse(
    r"\f g p k. p (\x y. k (f x) (g y))"
)
CHURCH_CURRY: Term = parse(r"\f x y. f (\p. p x y)")
CHURCH_UNCURRY: Term = parse(r"\f p. p f")
CHURCH_EVAL: Term = parse(r"\p. p (\f x. f x)")
CHURCH_INL: Term = parse(r"\x l r. l x")
CHURCH_INR: Term = parse(r"\y l r. r y")
CHURCH_FANIN: Term = parse(r"\f g e. e f g")
CHURCH_BIMAP_COPRODUCT: Term = parse(
    r"\f g e. e (\x l r. l (f x)) (\y l r. r (g y))"
)


@lam(term=CHURCH_IDENTITY)
def identity(value: A) -> A:
    """Identity morphism ``id: A -> A``."""
    return value


@lam(term=CHURCH_FST)
def fst(pair: tuple[A, B]) -> A:
    """First projection ``pi_1: A x B -> A``."""
    return pair[0]


@lam(term=CHURCH_SND)
def snd(pair: tuple[A, B]) -> B:
    """Second projection ``pi_2: A x B -> B``."""
    return pair[1]


@lam(term=CHURCH_FANOUT)
def fanout(
    first: Morphism[C, A],
    second: Morphism[C, B],
) -> Lambda[C, tuple[A, B]]:
    """Construct the pairing morphism ``<first, second>``."""
    first_lambda: Lambda[C, A] = ensure_unary(first, "fanout first argument")
    second_lambda: Lambda[C, B] = ensure_unary(second, "fanout second argument")
    symbolic = App(App(CHURCH_FANOUT, first_lambda.term), second_lambda.term)

    @lam(term=symbolic.normalize())
    def paired(value: C) -> tuple[A, B]:
        return (first_lambda(value), second_lambda(value))

    return paired


@lam(term=CHURCH_BIMAP_PRODUCT)
def bimap_product(
    first: Morphism[A, C],
    second: Morphism[B, D],
) -> Lambda[tuple[A, B], tuple[C, D]]:
    """Map two morphisms over a product independently."""
    first_lambda: Lambda[A, C] = ensure_unary(
        first, "bimap_product first argument"
    )
    second_lambda: Lambda[B, D] = ensure_unary(
        second, "bimap_product second argument"
    )
    symbolic = App(
        App(CHURCH_BIMAP_PRODUCT, first_lambda.term),
        second_lambda.term,
    )

    @lam(term=symbolic.normalize())
    def product_map(pair: tuple[A, B]) -> tuple[C, D]:
        return (first_lambda(pair[0]), second_lambda(pair[1]))

    return product_map


@lam(term=CHURCH_CURRY)
def curry(
    function: Morphism[tuple[A, B], C],
) -> Lambda[A, Lambda[B, C]]:
    """Curry ``(A x B -> C)`` into ``(A -> B -> C)``."""
    function_lambda: Lambda[tuple[A, B], C] = ensure_unary(function, "curry")
    symbolic = App(CHURCH_CURRY, function_lambda.term).normalize()

    @lam(term=symbolic)
    def curried(first: A, second: B) -> C:
        return function_lambda((first, second))

    return curried


@lam(term=CHURCH_UNCURRY)
def uncurry(
    function: Lambda[A, Lambda[B, C]] | Callable[[A, B], C],
) -> Lambda[tuple[A, B], C]:
    """Uncurry a fixed binary Lambda into ``(A x B -> C)``."""
    function_lambda = require_binary_lambda(function, "uncurry")
    symbolic = App(CHURCH_UNCURRY, function_lambda.term).normalize()

    @lam(term=symbolic)
    def uncurried(pair: tuple[A, B]) -> C:
        return function_lambda(pair[0])(pair[1])

    return uncurried


@lam(term=CHURCH_EVAL)
def eval_morphism(function_and_argument: tuple[Callable[[A], B], A]) -> B:
    """Evaluation morphism ``eval: (B^A x A) -> B``."""
    function, argument = function_and_argument
    return function(argument)


@lam(term=CHURCH_INL)
def inl(value: A) -> Either[A, Any]:
    """Left canonical injection into a coproduct."""
    return Left(value)


@lam(term=CHURCH_INR)
def inr(value: B) -> Either[Any, B]:
    """Right canonical injection into a coproduct."""
    return Right(value)


@lam(term=CHURCH_FANIN)
def fanin(
    left: Morphism[A, C],
    right: Morphism[B, C],
) -> Lambda[Either[A, B], C]:
    """Construct coproduct case analysis ``[left, right]``."""
    left_lambda: Lambda[A, C] = ensure_unary(left, "fanin first argument")
    right_lambda: Lambda[B, C] = ensure_unary(right, "fanin second argument")
    symbolic = App(App(CHURCH_FANIN, left_lambda.term), right_lambda.term)

    @lam(term=symbolic.normalize())
    def case_analysis(either: Either[A, B]) -> C:
        if isinstance(either, Left):
            return left_lambda(either.value)
        if isinstance(either, Right):
            return right_lambda(either.value)
        raise TypeError(
            "fanin expects an Either (Left or Right), "
            f"got {type(either).__name__}"
        )

    return case_analysis


@lam(term=CHURCH_BIMAP_COPRODUCT)
def bimap_coproduct(
    left: Morphism[A, C],
    right: Morphism[B, D],
) -> Lambda[Either[A, B], Either[C, D]]:
    """Map two morphisms over the variants of a coproduct."""
    left_lambda: Lambda[A, C] = ensure_unary(
        left, "bimap_coproduct first argument"
    )
    right_lambda: Lambda[B, D] = ensure_unary(
        right, "bimap_coproduct second argument"
    )
    symbolic = App(
        App(CHURCH_BIMAP_COPRODUCT, left_lambda.term),
        right_lambda.term,
    )

    @lam(term=symbolic.normalize())
    def coproduct_map(either: Either[A, B]) -> Either[C, D]:
        if isinstance(either, Left):
            return Left(left_lambda(either.value))
        if isinstance(either, Right):
            return Right(right_lambda(either.value))
        raise TypeError(
            f"bimap_coproduct expects an Either, got {type(either).__name__}"
        )

    return coproduct_map
