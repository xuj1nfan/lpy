"""Static assignments that exercise the public generic API."""

from lpy import Lambda, Maybe, Some, bind, curry, fanout, fmap, identity, lam, uncurry


def increment_impl(value: int) -> int:
    return value + 1


def render_impl(value: int) -> str:
    return str(value)


def add_impl(first: int, second: int) -> int:
    return first + second


def add_pair_impl(pair: tuple[int, int]) -> int:
    return pair[0] + pair[1]


increment: Lambda[int, int] = lam(increment_impl)
render: Lambda[int, str] = lam(render_impl)
add: Lambda[int, Lambda[int, int]] = lam(add_impl)
pair_add: Lambda[tuple[int, int], int] = lam(add_pair_impl)

render_incremented: Lambda[int, str] = render @ increment
increment_after_identity: Lambda[int, int] = increment @ identity
identity_after_increment: Lambda[int, int] = identity @ increment
identity_result: int = identity(1)
paired: Lambda[int, tuple[int, str]] = fanout(increment)(render)
curried: Lambda[int, Lambda[int, int]] = curry(pair_add)
uncurried: Lambda[tuple[int, int], int] = uncurry(add)
mapped: Maybe[int] = fmap(increment)(Some(1))
bound: Maybe[int] = bind(Some(1))(lambda value: Some(value + 1))
