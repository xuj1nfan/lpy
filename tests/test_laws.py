"""Property-based checks for the laws advertised by the public API."""

from hypothesis import given
from hypothesis import strategies as st

from lpy import (
    Env,
    Lens,
    Nothing,
    Some,
    bind,
    curry,
    duplicate,
    extract,
    fmap,
    identity,
    lam,
    maybe_to_list,
    pure_maybe,
    uncurry,
)


@given(st.integers())
def test_maybe_functor_laws(value: int) -> None:
    maybe = Some(value)
    increment = lam(lambda item: item + 1)
    double = lam(lambda item: item * 2)

    assert fmap(identity)(maybe) == maybe
    assert fmap(double @ increment)(maybe) == fmap(double)(fmap(increment)(maybe))
    assert fmap(identity)(Nothing) is Nothing


@given(st.lists(st.integers(), max_size=30))
def test_list_functor_laws(values: list[int]) -> None:
    increment = lam(lambda item: item + 1)
    square = lam(lambda item: item * item)

    assert fmap(identity)(values) == values
    assert fmap(square @ increment)(values) == fmap(square)(fmap(increment)(values))


@given(st.integers())
def test_maybe_monad_laws(value: int) -> None:
    first = lam(lambda item: Some(item + 1))
    second = lam(lambda item: Some(item * 2) if item >= 0 else Nothing)
    maybe = Some(value)

    assert bind(pure_maybe(value))(first) == first(value)
    assert bind(maybe)(pure_maybe) == maybe
    left = bind(bind(maybe)(first))(second)
    right = bind(maybe)(lam(lambda item: bind(first(item))(second)))
    assert left == right


@given(st.text(max_size=20), st.integers())
def test_env_comonad_laws(environment: str, value: int) -> None:
    comonad = Env(environment, value)
    duplicated = duplicate(comonad)

    assert extract(duplicated) == comonad
    assert fmap(extract)(duplicated) == comonad
    assert duplicate(duplicated) == fmap(duplicate)(duplicated)


@given(st.integers(), st.integers(), st.text(max_size=20))
def test_lens_laws(value: int, replacement: int, label: str) -> None:
    lens = Lens(
        lambda source: source["value"],
        lambda source, focus: {**source, "value": focus},
    )
    source = {"value": value, "label": label}

    assert lens.set(source)(lens.get(source)) == source
    assert lens.get(lens.set(source)(replacement)) == replacement
    assert lens.set(lens.set(source)(value + 1))(replacement) == lens.set(source)(
        replacement
    )


@given(st.integers(), st.integers())
def test_curry_uncurry_round_trip(first: int, second: int) -> None:
    add_pair = lam(lambda pair: pair[0] + pair[1])
    round_trip = uncurry(curry(add_pair))
    assert round_trip((first, second)) == add_pair((first, second))


@given(st.one_of(st.integers().map(Some), st.just(Nothing)))
def test_maybe_to_list_naturality(maybe: object) -> None:
    increment = lam(lambda value: value + 1)
    assert maybe_to_list.naturality_holds(increment, maybe)
