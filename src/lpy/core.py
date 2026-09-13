"""Core terms, parsing, printing, substitution, and reduction."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*'*\Z")


class LambdaError(Exception):
    """Base class for errors raised by :mod:`lpy`."""


class ParseError(LambdaError, ValueError):
    """Raised when a source string is not a valid lambda term."""

    def __init__(self, message: str, position: int, source: str | None = None):
        self.position = position
        if source is None:
            self.line = 1
            self.column = position + 1
        else:
            self.line = source.count("\n", 0, position) + 1
            line_start = source.rfind("\n", 0, position) + 1
            self.column = position - line_start + 1
        super().__init__(f"{message} at line {self.line}, column {self.column}")


class ReductionLimitExceeded(LambdaError, RuntimeError):
    """Raised when a bounded reduction has more steps remaining."""

    def __init__(self, term: Term, steps: int, strategy: Strategy, eta: bool):
        self.term = term
        self.steps = steps
        self.strategy = strategy
        self.eta = eta
        super().__init__(
            f"reduction limit reached after {steps} steps "
            f"(strategy={strategy.value}, eta={eta})"
        )


class Strategy(str, Enum):
    """Strong reduction traversal strategies."""

    NORMAL_ORDER = "normal"
    APPLICATIVE_ORDER = "applicative"


@dataclass(frozen=True, slots=True)
class _Free:
    name: str


@dataclass(frozen=True, slots=True)
class _Bound:
    index: int


@dataclass(frozen=True, slots=True)
class _Abs:
    body: _Node
    hint: str = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True, slots=True)
class _App:
    function: _Node
    argument: _Node


@dataclass(frozen=True, slots=True, eq=False)
class _Host:
    value: object = field(repr=False)
    label: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Host) and self.value is other.value

    def __hash__(self) -> int:
        return hash((_Host, id(self.value)))


_Node = _Free | _Bound | _Abs | _App | _Host


@dataclass(frozen=True, slots=True)
class DBFree:
    name: str


@dataclass(frozen=True, slots=True)
class DBBound:
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError("a De Bruijn index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DBAbstraction:
    body: DBTerm


@dataclass(frozen=True, slots=True)
class DBApplication:
    function: DBTerm
    argument: DBTerm


@dataclass(frozen=True, slots=True, eq=False)
class DBHost:
    label: str
    value: object = field(repr=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DBHost) and self.value is other.value

    def __hash__(self) -> int:
        return hash((DBHost, id(self.value)))


DBTerm = DBFree | DBBound | DBAbstraction | DBApplication | DBHost


@dataclass(frozen=True, slots=True)
class Term:
    """An immutable lambda term.

    Instances are created with :func:`Var`, :func:`Abs`, :func:`App`, or
    :func:`parse`.  Equality and hashing are alpha-equivalence aware.
    """

    _node: _Node = field(repr=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Term) and self._node == other._node

    def __hash__(self) -> int:
        return hash(self._node)

    def __str__(self) -> str:
        return _pretty(self._node)

    def __repr__(self) -> str:
        return f"Term({str(self)!r})"

    def __call__(self, argument: Any) -> Term:
        """Construct a symbolic application.

        Plain Python values are deliberately rejected.  A ``Lambda`` wrapper
        exposes its symbolic term through a ``term`` attribute and is accepted
        without importing the runtime layer here.
        """
        if isinstance(argument, Term):
            return App(self, argument)
        symbolic = getattr(argument, "term", None)
        if isinstance(symbolic, Term):
            return App(self, symbolic)
        raise TypeError(
            "a Term can only be applied to another Term or Lambda; "
            "wrap Python values with Host()"
        )

    def alpha_equivalent(self, other: Term) -> bool:
        return isinstance(other, Term) and self == other

    def substitute(self, variable: str, replacement: Term) -> Term:
        """Capture-avoiding substitution for a free variable."""
        _validate_name(variable)
        if not isinstance(replacement, Term):
            raise TypeError("replacement must be a Term")
        return Term(_subst_free(self._node, variable, replacement._node))

    def reduce_once(
        self, strategy: Strategy = Strategy.NORMAL_ORDER, *, eta: bool = False
    ) -> Term | None:
        strategy = _validate_strategy(strategy)
        result = _reduce_once(self._node, strategy, eta)
        return None if result is None else Term(result)

    def trace(
        self,
        strategy: Strategy = Strategy.NORMAL_ORDER,
        *,
        eta: bool = False,
        max_steps: int | None = None,
    ) -> Iterator[Term]:
        """Yield this term and each successive one-step reduction."""
        strategy = _validate_strategy(strategy)
        _validate_limit(max_steps)
        current = self
        steps = 0
        yield current
        while True:
            nxt = current.reduce_once(strategy, eta=eta)
            if nxt is None:
                return
            if max_steps is not None and steps >= max_steps:
                raise ReductionLimitExceeded(current, steps, strategy, eta)
            current = nxt
            steps += 1
            yield current

    def normalize(
        self,
        strategy: Strategy = Strategy.NORMAL_ORDER,
        *,
        eta: bool = False,
        max_steps: int | None = None,
    ) -> Term:
        """Reduce to a beta (or optionally beta-eta) normal form."""
        strategy = _validate_strategy(strategy)
        _validate_limit(max_steps)
        current = self
        steps = 0
        while True:
            nxt = current.reduce_once(strategy, eta=eta)
            if nxt is None:
                return current
            if max_steps is not None and steps >= max_steps:
                raise ReductionLimitExceeded(current, steps, strategy, eta)
            current = nxt
            steps += 1

    def is_normal_form(self, *, eta: bool = False) -> bool:
        return self.reduce_once(Strategy.NORMAL_ORDER, eta=eta) is None

    def to_debruijn(self) -> DBTerm:
        return _to_db(self._node)

    @property
    def abstraction_arity(self) -> int:
        """Number of consecutive outer abstractions."""
        node = self._node
        arity = 0
        while isinstance(node, _Abs):
            arity += 1
            node = node.body
        return arity

    @property
    def contains_host(self) -> bool:
        """Whether this term contains an opaque Python value."""
        return _contains_host(self._node)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"invalid variable name: {name!r}")


def Var(name: str) -> Term:
    """Construct a free variable."""
    _validate_name(name)
    return Term(_Free(name))


def Abs(parameter: str, body: Term) -> Term:
    """Construct an abstraction, binding free occurrences of *parameter*."""
    _validate_name(parameter)
    if not isinstance(body, Term):
        raise TypeError("body must be a Term")
    return Term(_Abs(_abstract(body._node, parameter, 0), parameter))


def App(function: Term, argument: Term) -> Term:
    """Construct an application."""
    if not isinstance(function, Term) or not isinstance(argument, Term):
        raise TypeError("function and argument must be Terms")
    return Term(_App(function._node, argument._node))


def Host(value: object, label: str | None = None) -> Term:
    """Create an irreducible symbolic reference to a Python value.

    Normalization never calls or otherwise evaluates the wrapped value.
    Host identity, rather than Python value equality, determines term equality.
    """
    if label is None:
        if callable(value):
            label = getattr(value, "__qualname__", getattr(value, "__name__", None))
        if label is None:
            simple = (str, bytes, int, float, complex, bool, type(None))
            label = repr(value) if isinstance(value, simple) else type(value).__name__
    if not isinstance(label, str) or not label:
        raise ValueError("a Host label must be a non-empty string")
    return Term(_Host(value, label.replace("]", "\\]")))


def _abstract(node: _Node, name: str, depth: int) -> _Node:
    if isinstance(node, _Free):
        return _Bound(depth) if node.name == name else node
    if isinstance(node, _Bound):
        return node
    if isinstance(node, _Host):
        return node
    if isinstance(node, _App):
        return _App(_abstract(node.function, name, depth), _abstract(node.argument, name, depth))
    return _Abs(_abstract(node.body, name, depth + 1), node.hint)


def _subst_free(node: _Node, name: str, replacement: _Node) -> _Node:
    if isinstance(node, _Free):
        return replacement if node.name == name else node
    if isinstance(node, _Bound):
        return node
    if isinstance(node, _Host):
        return node
    if isinstance(node, _Abs):
        return _Abs(_subst_free(node.body, name, replacement), node.hint)
    return _App(
        _subst_free(node.function, name, replacement),
        _subst_free(node.argument, name, replacement),
    )


def _shift(node: _Node, amount: int, cutoff: int = 0) -> _Node:
    if isinstance(node, _Bound):
        return _Bound(node.index + amount) if node.index >= cutoff else node
    if isinstance(node, (_Free, _Host)):
        return node
    if isinstance(node, _Abs):
        return _Abs(_shift(node.body, amount, cutoff + 1), node.hint)
    return _App(_shift(node.function, amount, cutoff), _shift(node.argument, amount, cutoff))


def _subst_bound(node: _Node, index: int, replacement: _Node) -> _Node:
    if isinstance(node, _Bound):
        return replacement if node.index == index else node
    if isinstance(node, (_Free, _Host)):
        return node
    if isinstance(node, _Abs):
        return _Abs(_subst_bound(node.body, index + 1, _shift(replacement, 1)), node.hint)
    return _App(
        _subst_bound(node.function, index, replacement),
        _subst_bound(node.argument, index, replacement),
    )


def _beta(abs_node: _Abs, argument: _Node) -> _Node:
    return _shift(_subst_bound(abs_node.body, 0, _shift(argument, 1)), -1)


def _bound_occurs(node: _Node, target: int, depth: int = 0) -> bool:
    if isinstance(node, _Bound):
        return node.index == target + depth
    if isinstance(node, (_Free, _Host)):
        return False
    if isinstance(node, _Abs):
        return _bound_occurs(node.body, target, depth + 1)
    return _bound_occurs(node.function, target, depth) or _bound_occurs(
        node.argument, target, depth
    )


def _eta(node: _Abs) -> _Node | None:
    if not isinstance(node.body, _App) or not isinstance(node.body.argument, _Bound):
        return None
    if node.body.argument.index != 0 or _bound_occurs(node.body.function, 0):
        return None
    return _shift(node.body.function, -1)


def _reduce_once(node: _Node, strategy: Strategy, eta: bool) -> _Node | None:
    if strategy is Strategy.NORMAL_ORDER:
        if isinstance(node, _App) and isinstance(node.function, _Abs):
            return _beta(node.function, node.argument)
        if eta and isinstance(node, _Abs):
            reduced = _eta(node)
            if reduced is not None:
                return reduced
        if isinstance(node, _Abs):
            body = _reduce_once(node.body, strategy, eta)
            return None if body is None else _Abs(body, node.hint)
        if isinstance(node, _App):
            function = _reduce_once(node.function, strategy, eta)
            if function is not None:
                return _App(function, node.argument)
            argument = _reduce_once(node.argument, strategy, eta)
            return None if argument is None else _App(node.function, argument)
        return None

    if isinstance(node, _Abs):
        body = _reduce_once(node.body, strategy, eta)
        if body is not None:
            return _Abs(body, node.hint)
        if eta:
            return _eta(node)
        return None
    if isinstance(node, _App):
        function = _reduce_once(node.function, strategy, eta)
        if function is not None:
            return _App(function, node.argument)
        argument = _reduce_once(node.argument, strategy, eta)
        if argument is not None:
            return _App(node.function, argument)
        if isinstance(node.function, _Abs):
            return _beta(node.function, node.argument)
    return None


def _validate_strategy(strategy: Strategy) -> Strategy:
    if not isinstance(strategy, Strategy):
        raise TypeError("strategy must be a Strategy value")
    return strategy


def _validate_limit(max_steps: int | None) -> None:
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError("max_steps must be a non-negative integer or None")


def _to_db(node: _Node) -> DBTerm:
    if isinstance(node, _Free):
        return DBFree(node.name)
    if isinstance(node, _Bound):
        return DBBound(node.index)
    if isinstance(node, _Host):
        return DBHost(node.label, node.value)
    if isinstance(node, _Abs):
        return DBAbstraction(_to_db(node.body))
    return DBApplication(_to_db(node.function), _to_db(node.argument))


def _free_names(node: _Node) -> set[str]:
    if isinstance(node, _Free):
        return {node.name}
    if isinstance(node, _Bound):
        return set()
    if isinstance(node, _Host):
        return set()
    if isinstance(node, _Abs):
        return _free_names(node.body)
    return _free_names(node.function) | _free_names(node.argument)


def _fresh(base: str, used: set[str]) -> str:
    candidate = base
    while candidate in used:
        candidate += "'"
    return candidate


def _pretty(node: _Node) -> str:
    return _render(node, (), _free_names(node))


def _render(node: _Node, env: tuple[str, ...], used: set[str]) -> str:
    if isinstance(node, _Free):
        return node.name
    if isinstance(node, _Bound):
        return env[-1 - node.index] if node.index < len(env) else f"#${node.index}"
    if isinstance(node, _Host):
        return f"@python[{node.label}]"
    if isinstance(node, _Abs):
        name = _fresh(node.hint, used)
        body = _render(node.body, (*env, name), used | {name})
        return f"\\{name}. {body}"
    left = _render(node.function, env, used)
    right = _render(node.argument, env, used)
    if isinstance(node.function, _Abs):
        left = f"({left})"
    if isinstance(node.argument, (_Abs, _App)):
        right = f"({right})"
    return f"{left} {right}"


def _contains_host(node: _Node) -> bool:
    if isinstance(node, _Host):
        return True
    if isinstance(node, (_Free, _Bound)):
        return False
    if isinstance(node, _Abs):
        return _contains_host(node.body)
    return _contains_host(node.function) or _contains_host(node.argument)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    value: str
    position: int


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    while i < len(source):
        if source[i].isspace():
            i += 1
            continue
        if source[i] in r"\().":
            tokens.append(_Token(source[i], source[i], i))
            i += 1
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*'*", source[i:])
        if match:
            value = match.group(0)
            tokens.append(_Token("ID", value, i))
            i += len(value)
            continue
        raise ParseError(f"unexpected character {source[i]!r}", i, source)
    tokens.append(_Token("EOF", "", len(source)))
    return tokens


class _Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = _tokenize(source)
        self.index = 0

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def advance(self) -> _Token:
        token = self.current
        self.index += 1
        return token

    def expect(self, kind: str) -> _Token:
        if self.current.kind != kind:
            raise ParseError(
                f"expected {kind!r}, got {self.current.kind!r}",
                self.current.position,
                self.source,
            )
        return self.advance()

    def parse(self) -> Term:
        if self.current.kind == "EOF":
            raise ParseError("expected a term", self.current.position, self.source)
        result = self.parse_term()
        self.expect("EOF")
        return result

    def parse_term(self) -> Term:
        if self.current.kind == "\\":
            self.advance()
            parameters: list[str] = []
            while self.current.kind == "ID":
                parameters.append(self.advance().value)
            if not parameters:
                raise ParseError(
                    "expected an abstraction parameter", self.current.position, self.source
                )
            self.expect(".")
            body = self.parse_term()
            for parameter in reversed(parameters):
                body = Abs(parameter, body)
            return body
        return self.parse_application()

    def parse_application(self) -> Term:
        result = self.parse_atom()
        while self.current.kind in {"ID", "("}:
            result = App(result, self.parse_atom())
        return result

    def parse_atom(self) -> Term:
        if self.current.kind == "ID":
            return Var(self.advance().value)
        if self.current.kind == "(":
            self.advance()
            result = self.parse_term()
            self.expect(")")
            return result
        raise ParseError(
            "expected a variable or parenthesized term", self.current.position, self.source
        )


def parse(source: str) -> Term:
    """Parse an ASCII lambda expression."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    return _Parser(source).parse()
