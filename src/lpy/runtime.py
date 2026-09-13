"""Python-callable, curried lambda values with a symbolic calculus view."""

from __future__ import annotations

import ast
import functools
import inspect
import operator
import textwrap
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, cast, overload

from .core import Abs, App, Host, Term, Var, _free_names


class SymbolicKind(str, Enum):
    """How faithfully a :class:`Lambda` could be represented symbolically."""

    PURE = "pure"
    EXTENDED = "extended"
    OPAQUE = "opaque"


class SymbolicAnalysisError(Exception):
    """Internal signal that a Python body cannot be represented safely."""


InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
SourceT = TypeVar("SourceT")
MiddleT = TypeVar("MiddleT")
TargetT = TypeVar("TargetT")
SecondInputT = TypeVar("SecondInputT")
ThirdInputT = TypeVar("ThirdInputT")


class _Composable(Protocol[InputT_contra, OutputT_co]):
    """Static shape accepted on the right of Lambda composition."""

    @property
    def term(self) -> Term: ...

    @property
    def remaining_arity(self) -> int: ...

    def __call__(self, argument: InputT_contra, /) -> OutputT_co: ...


class Lambda(Generic[InputT_contra, OutputT_co]):
    """A strictly curried Python callable with a symbolic lambda term."""

    def __init__(
        self,
        function: Callable[..., Any],
        parameters: tuple[inspect.Parameter, ...],
        term: Term,
        symbolic_kind: SymbolicKind,
        symbolic_reason: str | None = None,
        bound_arguments: tuple[Any, ...] = (),
    ) -> None:
        self._function = function
        self._parameters = parameters
        self._term = term
        self._symbolic_kind = symbolic_kind
        self._symbolic_reason = symbolic_reason
        self._bound_arguments = bound_arguments
        functools.update_wrapper(self, function, updated=())
        if not hasattr(self, "__name__"):
            self.__name__ = type(function).__name__
        if not hasattr(self, "__qualname__"):
            self.__qualname__ = self.__name__
        self.__signature__ = self._next_signature()

    @property
    def term(self) -> Term:
        """The safe, non-executing symbolic view of this callable."""
        return self._term

    @property
    def arity(self) -> int:
        return len(self._parameters)

    @property
    def remaining_arity(self) -> int:
        return self.arity - len(self._bound_arguments)

    @property
    def symbolic_kind(self) -> SymbolicKind:
        return self._symbolic_kind

    @property
    def symbolic_reason(self) -> str | None:
        return self._symbolic_reason

    def _next_signature(self) -> inspect.Signature:
        parameter = self._parameters[len(self._bound_arguments)]
        parameter = parameter.replace(
            kind=inspect.Parameter.POSITIONAL_ONLY,
            default=inspect.Parameter.empty,
        )
        return inspect.Signature((parameter,))

    @overload
    def __call__(self, argument: InputT_contra, /) -> OutputT_co: ...

    @overload
    def __call__(self, argument: Term, /) -> Term: ...

    def __call__(self, *arguments: Any, **keywords: Any) -> Any:
        if keywords or len(arguments) != 1:
            raise TypeError(
                f"{self.__name__} is strictly curried and accepts exactly "
                "one positional argument per call"
            )

        argument = arguments[0]
        if isinstance(argument, Term):
            return App(self._term, argument)

        next_arguments = (*self._bound_arguments, argument)
        if len(next_arguments) == self.arity:
            return self._function(*next_arguments)

        argument_term, argument_kind, argument_reason = _symbolic_argument(argument)
        kind = _combine_kind(self._symbolic_kind, argument_kind)
        reason = self._symbolic_reason or argument_reason
        return Lambda(
            self._function,
            self._parameters,
            App(self._term, argument_term),
            kind,
            reason,
            next_arguments,
        )

    def __matmul__(
        self: Lambda[MiddleT, TargetT],
        other: _Composable[SourceT, MiddleT],
    ) -> Lambda[SourceT, TargetT]:
        if not isinstance(other, Lambda):
            return NotImplemented
        if self.remaining_arity != 1 or other.remaining_arity != 1:
            raise TypeError("function composition requires two unary Lambda values")

        used = _free_names(self.term._node) | _free_names(other.term._node)
        parameter = "_compose_value"
        while parameter in used:
            parameter += "_"
        variable = Var(parameter)
        composed_term = Abs(parameter, App(self.term, App(other.term, variable)))

        def composed(value: Any) -> Any:
            return self(other(value))

        composed.__name__ = f"{self.__name__}_after_{other.__name__}"
        composed.__qualname__ = composed.__name__
        composed.__doc__ = f"Composition of {self.__name__} after {other.__name__}."
        parameter_info = inspect.Parameter(
            parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        kind = _combine_kind(self.symbolic_kind, other.symbolic_kind)
        reason = self.symbolic_reason or other.symbolic_reason
        return Lambda(composed, (parameter_info,), composed_term, kind, reason)

    def __str__(self) -> str:
        return str(self.term)

    def __repr__(self) -> str:
        return (
            f"Lambda(name={self.__name__!r}, remaining_arity={self.remaining_arity}, "
            f"symbolic_kind={self.symbolic_kind.value!r})"
        )


def _symbolic_argument(value: Any) -> tuple[Term, SymbolicKind, str | None]:
    if isinstance(value, Lambda):
        return value.term, value.symbolic_kind, value.symbolic_reason
    return Host(value), SymbolicKind.EXTENDED, None


def _combine_kind(left: SymbolicKind, right: SymbolicKind) -> SymbolicKind:
    if SymbolicKind.OPAQUE in (left, right):
        return SymbolicKind.OPAQUE
    if SymbolicKind.EXTENDED in (left, right):
        return SymbolicKind.EXTENDED
    return SymbolicKind.PURE


def _validated_parameters(function: Callable[..., Any]) -> tuple[inspect.Parameter, ...]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise TypeError("lam() requires a callable with an inspectable signature") from error

    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise TypeError("a Lambda must have at least one parameter")
    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                "lam() only accepts fixed positional parameters; "
                f"{parameter.name!r} is {parameter.kind.description}"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError("lam() does not accept parameters with default values")
    return parameters


def _opaque_term(
    function: Callable[..., Any], parameters: tuple[inspect.Parameter, ...]
) -> Term:
    label = getattr(
        function,
        "__qualname__",
        getattr(function, "__name__", type(function).__name__),
    )
    body = Host(function, label)
    for parameter in parameters:
        body = App(body, Var(parameter.name))
    for parameter in reversed(parameters):
        body = Abs(parameter.name, body)
    return body


_BINARY_OPERATORS: dict[type[ast.operator], tuple[Callable[..., Any], str]] = {
    ast.Add: (operator.add, "operator.add"),
    ast.Sub: (operator.sub, "operator.sub"),
    ast.Mult: (operator.mul, "operator.mul"),
    ast.MatMult: (operator.matmul, "operator.matmul"),
    ast.Div: (operator.truediv, "operator.truediv"),
    ast.FloorDiv: (operator.floordiv, "operator.floordiv"),
    ast.Mod: (operator.mod, "operator.mod"),
    ast.Pow: (operator.pow, "operator.pow"),
    ast.LShift: (operator.lshift, "operator.lshift"),
    ast.RShift: (operator.rshift, "operator.rshift"),
    ast.BitOr: (operator.or_, "operator.or_"),
    ast.BitXor: (operator.xor, "operator.xor"),
    ast.BitAnd: (operator.and_, "operator.and_"),
}

_UNARY_OPERATORS: dict[type[ast.unaryop], tuple[Callable[..., Any], str]] = {
    ast.UAdd: (operator.pos, "operator.pos"),
    ast.USub: (operator.neg, "operator.neg"),
    ast.Not: (operator.not_, "operator.not_"),
    ast.Invert: (operator.invert, "operator.invert"),
}


def _not_contains(container: Any, item: Any) -> bool:
    return item not in container

_COMPARISON_OPERATORS: dict[type[ast.cmpop], tuple[Callable[..., Any], str]] = {
    ast.Eq: (operator.eq, "operator.eq"),
    ast.NotEq: (operator.ne, "operator.ne"),
    ast.Lt: (operator.lt, "operator.lt"),
    ast.LtE: (operator.le, "operator.le"),
    ast.Gt: (operator.gt, "operator.gt"),
    ast.GtE: (operator.ge, "operator.ge"),
    ast.Is: (operator.is_, "operator.is_"),
    ast.IsNot: (operator.is_not, "operator.is_not"),
    ast.In: (operator.contains, "operator.contains"),
    ast.NotIn: (_not_contains, "operator.not_contains"),
}


class _ExpressionCompiler:
    def __init__(self, bindings: dict[str, Term], namespace: dict[str, Any]):
        self.bindings = bindings
        self.namespace = namespace
        self.used_host = False

    def host(self, value: object, label: str) -> Term:
        self.used_host = True
        return Host(value, label)

    def compile(self, node: ast.AST) -> Term:
        if isinstance(node, ast.Name):
            if node.id in self.bindings:
                return self.bindings[node.id]
            if node.id in self.namespace:
                value = self.namespace[node.id]
                if isinstance(value, Lambda):
                    if value.symbolic_kind is not SymbolicKind.PURE:
                        self.used_host = True
                    return value.term
                if isinstance(value, Term):
                    if value.contains_host:
                        self.used_host = True
                    return value
                return self.host(value, node.id)
            raise SymbolicAnalysisError(f"unresolved name {node.id!r}")

        if isinstance(node, ast.Lambda):
            names = _ast_lambda_parameters(node)
            nested = dict(self.bindings)
            nested.update((name, Var(name)) for name in names)
            compiler = _ExpressionCompiler(nested, self.namespace)
            body = compiler.compile(node.body)
            self.used_host |= compiler.used_host
            for name in reversed(names):
                body = Abs(name, body)
            return body

        if isinstance(node, ast.Call):
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                raise SymbolicAnalysisError("keyword and starred calls are not symbolic")
            result = self.compile(node.func)
            for argument in node.args:
                result = App(result, self.compile(argument))
            return result

        if isinstance(node, ast.Constant):
            return self.host(node.value, repr(node.value))

        if isinstance(node, ast.BinOp):
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise SymbolicAnalysisError(f"unsupported operator {type(node.op).__name__}")
            function, label = operation
            return App(
                App(self.host(function, label), self.compile(node.left)),
                self.compile(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            operation = _UNARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise SymbolicAnalysisError(f"unsupported operator {type(node.op).__name__}")
            function, label = operation
            return App(self.host(function, label), self.compile(node.operand))

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise SymbolicAnalysisError("chained comparisons are not symbolic")
            operation = _COMPARISON_OPERATORS.get(type(node.ops[0]))
            if operation is None:
                raise SymbolicAnalysisError(
                    f"unsupported comparison {type(node.ops[0]).__name__}"
                )
            function, label = operation
            left = self.compile(node.left)
            right = self.compile(node.comparators[0])
            if isinstance(node.ops[0], (ast.In, ast.NotIn)):
                left, right = right, left
            return App(App(self.host(function, label), left), right)

        if isinstance(node, ast.Attribute):
            return App(
                App(self.host(getattr, "getattr"), self.compile(node.value)),
                self.host(node.attr, repr(node.attr)),
            )

        if isinstance(node, ast.Subscript):
            return App(
                App(
                    self.host(operator.getitem, "operator.getitem"),
                    self.compile(node.value),
                ),
                self.compile(node.slice),
            )

        if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError) as error:
                raise SymbolicAnalysisError("non-literal collection") from error
            return self.host(value, repr(value))

        raise SymbolicAnalysisError(
            f"unsupported expression {type(node).__name__}"
        )


def _ast_lambda_parameters(node: ast.Lambda) -> tuple[str, ...]:
    arguments = node.args
    if (
        arguments.defaults
        or arguments.kw_defaults
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
    ):
        raise SymbolicAnalysisError("nested lambda has an unsupported signature")
    return tuple(arg.arg for arg in (*arguments.posonlyargs, *arguments.args))


def _expression_from_source(
    function: Callable[..., Any], parameter_names: tuple[str, ...]
) -> ast.expr:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as error:
        raise SymbolicAnalysisError("Python source is unavailable") from error
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SymbolicAnalysisError("Python source could not be parsed") from error

    function_name = getattr(function, "__name__", None)
    if function_name and function_name != "<lambda>":
        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
        if len(candidates) != 1:
            raise SymbolicAnalysisError("could not identify the function definition")
        body = list(candidates[0].body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            raise SymbolicAnalysisError("only a single return expression is symbolic")
        return body[0].value

    lambda_candidates: list[ast.Lambda] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Lambda):
            continue
        try:
            names = _ast_lambda_parameters(node)
        except SymbolicAnalysisError:
            continue
        if names == parameter_names:
            lambda_candidates.append(node)
    if len(lambda_candidates) != 1:
        raise SymbolicAnalysisError("could not uniquely identify the lambda expression")
    return lambda_candidates[0].body


def _namespace(function: Callable[..., Any]) -> dict[str, Any]:
    try:
        closure = inspect.getclosurevars(function)
    except (TypeError, ValueError):
        return {}
    return {
        **closure.builtins,
        **closure.globals,
        **closure.nonlocals,
    }


def _analyze(
    function: Callable[..., Any], parameters: tuple[inspect.Parameter, ...]
) -> tuple[Term, SymbolicKind, str | None]:
    names = tuple(parameter.name for parameter in parameters)
    try:
        expression = _expression_from_source(function, names)
        compiler = _ExpressionCompiler(
            {name: Var(name) for name in names}, _namespace(function)
        )
        body = compiler.compile(expression)
        for name in reversed(names):
            body = Abs(name, body)
        kind = SymbolicKind.EXTENDED if compiler.used_host else SymbolicKind.PURE
        return body, kind, None
    except SymbolicAnalysisError as error:
        return _opaque_term(function, parameters), SymbolicKind.OPAQUE, str(error)


P = ParamSpec("P")
R = TypeVar("R")


class LambdaDecorator(Protocol):
    """Typing interface for ``@lam(...)`` decorator factories."""

    @overload
    def __call__(
        self,
        function: Callable[[InputT_contra], OutputT_co],
        /,
    ) -> Lambda[InputT_contra, OutputT_co]: ...

    @overload
    def __call__(
        self,
        function: Callable[[InputT_contra, SecondInputT], OutputT_co],
        /,
    ) -> Lambda[InputT_contra, Lambda[SecondInputT, OutputT_co]]: ...

    @overload
    def __call__(
        self,
        function: Callable[
            [InputT_contra, SecondInputT, ThirdInputT],
            OutputT_co,
        ],
        /,
    ) -> Lambda[
        InputT_contra,
        Lambda[SecondInputT, Lambda[ThirdInputT, OutputT_co]],
    ]: ...

    def __call__(self, function: Callable[P, R], /) -> Lambda[Any, R]: ...


@overload
def lam(
    function: Callable[
        [InputT_contra, SecondInputT, ThirdInputT],
        OutputT_co,
    ],
    *,
    term: Term | None = None,
) -> Lambda[
    InputT_contra,
    Lambda[SecondInputT, Lambda[ThirdInputT, OutputT_co]],
]: ...


@overload
def lam(
    function: Callable[[InputT_contra, SecondInputT], OutputT_co],
    *,
    term: Term | None = None,
) -> Lambda[InputT_contra, Lambda[SecondInputT, OutputT_co]]: ...


@overload
def lam(
    function: Callable[[InputT_contra], OutputT_co],
    *,
    term: Term | None = None,
) -> Lambda[InputT_contra, OutputT_co]: ...


@overload
def lam(
    function: Callable[P, R],
    *,
    term: Term | None = None,
) -> Lambda[Any, R]: ...


@overload
def lam(
    function: None = None,
    *,
    term: Term | None = None,
) -> LambdaDecorator: ...


def lam(
    function: Callable[..., Any] | None = None, *, term: Term | None = None
) -> Lambda[Any, Any] | LambdaDecorator:
    """Wrap a Python callable as a strictly curried :class:`Lambda`."""

    def decorate(target: Callable[P, R]) -> Lambda[Any, R]:
        if not callable(target):
            raise TypeError("lam() requires a callable")
        parameters = _validated_parameters(target)
        if term is not None:
            if not isinstance(term, Term):
                raise TypeError("term= must be a Term")
            if term.abstraction_arity < len(parameters):
                raise ValueError(
                    "term= must have at least one outer abstraction per "
                    "Python parameter"
                )
            kind = SymbolicKind.EXTENDED if term.contains_host else SymbolicKind.PURE
            symbolic_term, reason = term, None
        else:
            symbolic_term, kind, reason = _analyze(target, parameters)
        return Lambda(target, parameters, symbolic_term, kind, reason)

    if function is None:
        return cast(LambdaDecorator, decorate)
    return decorate(function)


__all__ = ["Lambda", "SymbolicKind", "lam"]
