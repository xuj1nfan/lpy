import inspect

import pytest

from lpy import (
    App,
    DBHost,
    Host,
    SymbolicKind,
    Var,
    lam,
    parse,
)


identity = lam(lambda x: x)
increment = lam(lambda x: x + 1)
double = lam(lambda x: x * 2)


@lam
def add(x, y):
    return x + y


@lam
def apply_function(function, value):
    return function(value)


@lam(term=parse(r"\x. \y. x"))
def first(x, y):
    return x


def test_python_execution_and_strict_currying():
    assert identity(3) == 3
    assert add(1)(2) == 3
    assert add(1).remaining_arity == 1
    assert str(inspect.signature(add)) == "(x, /)"
    assert str(inspect.signature(add(1))) == "(y, /)"
    with pytest.raises(TypeError):
        add(1, 2)
    with pytest.raises(TypeError):
        add(x=1)


def test_lambda_is_a_python_callback():
    assert list(map(increment, [1, 2, 3])) == [2, 3, 4]
    key = lam(lambda item: item[1])
    assert sorted([("b", 2), ("a", 1)], key=key) == [("a", 1), ("b", 2)]


def test_higher_order_execution_and_composition():
    assert apply_function(increment)(10) == 11
    increment_after_double = increment @ double
    assert increment_after_double(3) == 7
    assert increment_after_double.remaining_arity == 1


def test_symbolic_dispatch_does_not_execute_python():
    variable = Var("value")
    assert identity(variable) == App(identity.term, variable)
    assert Var("f")(identity) == App(Var("f"), identity.term)
    with pytest.raises(TypeError):
        Var("f")(3)


def test_symbolic_analysis_kinds():
    assert identity.symbolic_kind is SymbolicKind.PURE
    assert identity.term == parse(r"\x. x")
    assert increment.symbolic_kind is SymbolicKind.EXTENDED
    assert increment.term.contains_host
    assert first.symbolic_kind is SymbolicKind.PURE
    assert first.term == parse(r"\a. \b. a")


def test_opaque_body_is_not_run_during_decoration():
    events = []

    @lam
    def effectful(value):
        events.append(value)
        return value

    assert events == []
    assert effectful.symbolic_kind is SymbolicKind.OPAQUE
    assert effectful.symbolic_reason
    assert effectful(4) == 4
    assert events == [4]


def test_host_nodes_never_execute_during_normalization():
    calls = []

    def host_function(value):
        calls.append(value)
        return value

    term = App(Host(host_function), Host(1))
    assert term.normalize() == term
    assert calls == []
    assert isinstance(Host(host_function).to_debruijn(), DBHost)
    other_function = lambda value: value
    assert Host(host_function, "same") != Host(other_function, "same")
    assert Host(host_function, "same").to_debruijn() != Host(
        other_function, "same"
    ).to_debruijn()


def test_explicit_term_arity_and_signature_validation():
    with pytest.raises(ValueError):
        lam(lambda x, y: x, term=parse(r"\x. x"))
    with pytest.raises(TypeError):
        lam(lambda x=1: x)
    with pytest.raises(TypeError):
        lam(lambda *items: items)


def test_callable_results_are_not_automatically_wrapped():
    maker = lam(lambda value: lambda other: value + other)
    result = maker(2)
    assert not hasattr(result, "term")
    assert result(3) == 5


def test_callable_objects_and_builtins_can_be_wrapped():
    class Negate:
        def __call__(self, value):
            return -value

    negate = lam(Negate())
    absolute = lam(abs)
    assert negate(3) == -3
    assert absolute(-3) == 3
    assert negate.symbolic_kind is SymbolicKind.OPAQUE
