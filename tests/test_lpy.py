import pytest

from lpy import (
    Abs,
    App,
    DBAbstraction,
    DBApplication,
    DBBound,
    DBFree,
    ParseError,
    ReductionLimitExceeded,
    Strategy,
    Var,
    parse,
)


def test_parse_and_print_round_trip():
    term = parse(r"\x y. x (y x)")
    assert parse(str(term)) == term
    assert str(term) == r"\x. \y. x (y x)"


def test_alpha_equivalence_and_debruijn_view():
    left = parse(r"\x. x")
    right = parse(r"\z. z")
    assert left == right
    assert hash(left) == hash(right)
    assert left.to_debruijn() == DBAbstraction(DBBound(0))
    assert parse(r"x y").to_debruijn() == DBApplication(DBFree("x"), DBFree("y"))


def test_shadowing_is_lexical():
    assert str(parse(r"\x. \x. x")) == r"\x. \x'. x'"
    assert parse(r"\x. \x. x").to_debruijn() == DBAbstraction(
        DBAbstraction(DBBound(0))
    )


def test_capture_avoiding_substitution():
    result = parse(r"\y. x").substitute("x", Var("y"))
    assert str(result) == r"\y'. y"
    assert result.to_debruijn() == DBAbstraction(DBFree("y"))


def test_beta_reduction_and_strong_normalization():
    term = App(parse(r"\x. x"), Var("y"))
    assert term.reduce_once() == Var("y")
    assert term.normalize() == Var("y")
    assert parse(r"\x. (\y. y) x").normalize() == parse(r"\x. x")


def test_eta_reduction_side_condition():
    assert parse(r"\x. f x").normalize(eta=True) == Var("f")
    assert parse(r"\x. x x").normalize(eta=True) == parse(r"\x. x x")


def test_strategy_difference_and_limit():
    term = parse(r"(\x. z) ((\x. x x) (\x. x x))")
    assert term.normalize(Strategy.NORMAL_ORDER) == Var("z")
    with pytest.raises(ReductionLimitExceeded):
        term.normalize(Strategy.APPLICATIVE_ORDER, max_steps=10)


def test_trace_includes_initial_term():
    stages = list(App(parse(r"\x. x"), Var("a")).trace())
    assert stages == [App(parse(r"\x. x"), Var("a")), Var("a")]


def test_parser_errors():
    with pytest.raises(ParseError) as error:
        parse(r"\x x")
    assert error.value.position == 4


def test_invalid_limits_and_names():
    with pytest.raises(ValueError):
        Var("1x")
    with pytest.raises(ValueError):
        Var("x-")
    with pytest.raises(ValueError):
        parse(r"\x. x").normalize(max_steps=-1)
