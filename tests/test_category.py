from typing import Any

import pytest

from lpy import (
    CHURCH_FANIN,
    CHURCH_INL,
    CHURCH_INR,
    App,
    Env,
    Fix,
    Left,
    Lens,
    Nothing,
    Right,
    Some,
    Store,
    Var,
    ana,
    bimap_coproduct,
    bimap_product,
    bind,
    cata,
    curry,
    duplicate,
    either_to_maybe,
    eval_morphism,
    extend,
    extract,
    fanin,
    fanout,
    fmap,
    fst,
    hylo,
    identity,
    inl,
    inr,
    kleisli_compose,
    lam,
    list_to_maybe,
    maybe_to_list,
    parse,
    pure_maybe,
    snd,
    uncurry,
)
from lpy.category import (
    CHURCH_BIMAP_COPRODUCT,
    CHURCH_BIMAP_PRODUCT,
    CHURCH_CURRY,
    CHURCH_FANOUT,
    CHURCH_FST,
    CHURCH_SOME,
    CHURCH_UNCURRY,
    KLEISLI_COMPOSE_TERM,
    LIST_FUNCTOR,
    MAYBE_FUNCTOR,
    functor_for,
)


def test_category_identity_and_composition_laws():
    double = lam(lambda x: x * 2)
    add_one = lam(lambda x: x + 1)
    negate = lam(lambda x: -x)

    # 1. Identity laws: id @ f == f, f @ id == f
    id_after_double = identity @ double
    double_after_id = double @ identity
    assert id_after_double(5) == double(5) == 10
    assert double_after_id(5) == double(5) == 10

    # 2. Associativity law: (f @ g) @ h == f @ (g @ h)
    left_assoc = (negate @ add_one) @ double
    right_assoc = negate @ (add_one @ double)
    assert left_assoc(3) == right_assoc(3) == -7


def test_cartesian_products_and_universal_property():
    double = lam(lambda x: x * 2)
    add_one = lam(lambda x: x + 1)

    # Projections
    pair = (10, "hello")
    assert fst(pair) == 10
    assert snd(pair) == "hello"

    # Pairing / Fanout: <f, g>(x) = (f(x), g(x))
    paired = fanout(double)(add_one)
    assert paired(4) == (8, 5)

    # Universal property of product: fst @ <f, g> == f, snd @ <f, g> == g
    fst_paired = fst @ paired
    snd_paired = snd @ paired
    assert fst_paired(4) == double(4) == 8
    assert snd_paired(4) == add_one(4) == 5

    # Bimap product: (f x g)(x, y) = (f(x), g(y))
    prod_map = bimap_product(double)(add_one)
    assert prod_map((3, 10)) == (6, 11)


def test_currying_adjunction():
    # Hom(A x B, C) ~= Hom(A, C^B)
    add_pair = lam(lambda p: p[0] + p[1])
    curried_add = curry(add_pair)
    assert curried_add(3)(4) == 7

    # Round trip: uncurry(curry(f)) == f
    uncurried_again = uncurry(curried_add)
    assert uncurried_again((3, 4)) == 7

    # Round trip: curry(uncurry(g)) == g
    @lam
    def multiply(x, y):
        return x * y

    curried_again = curry(uncurry(multiply))
    assert curried_again(3)(4) == 12

    # Evaluation morphism: eval(f, x) = f(x)
    double = lam(lambda x: x * 2)
    assert eval_morphism((double, 21)) == 42


def test_coproducts_and_fanin_universal_property():
    double = lam(lambda x: x * 2)
    greet = lam(lambda name: f"Hello {name}")

    # Canonical injections
    left_val = inl(10)
    right_val = inr("World")
    assert isinstance(left_val, Left)
    assert left_val.value == 10
    assert isinstance(right_val, Right)
    assert right_val.value == "World"

    # Fanin / case analysis: [f, g]
    cases = fanin(double)(greet)
    assert cases(left_val) == 20
    assert cases(right_val) == "Hello World"

    # Universal property: [f, g] @ inl == f, [f, g] @ inr == g
    case_inl = cases @ inl
    case_inr = cases @ inr
    assert case_inl(7) == double(7) == 14
    assert case_inr("Bob") == greet("Bob") == "Hello Bob"

    # Bimap coproduct: f + g
    to_upper = lam(lambda s: s.upper())
    coprod_map = bimap_coproduct(double)(to_upper)
    assert coprod_map(inl(5)) == Left(10)
    assert coprod_map(inr("test")) == Right("TEST")


def test_functor_laws():
    double = lam(lambda x: x * 2)
    add_one = lam(lambda x: x + 1)

    # 1. Identity law: fmap(id)(x) == x
    assert fmap(identity)(Some(42)) == Some(42)
    assert fmap(identity)(Nothing) == Nothing
    assert fmap(identity)([1, 2, 3]) == [1, 2, 3]

    # 2. Composition law: fmap(f @ g) == fmap(f) @ fmap(g)
    comp_fn = double @ add_one
    fmap_comp = fmap(comp_fn)
    comp_fmap = fmap(double) @ fmap(add_one)

    target_maybe = Some(5)
    assert fmap_comp(target_maybe) == comp_fmap(target_maybe) == Some(12)

    target_list = [1, 2, 3]
    assert fmap_comp(target_list) == comp_fmap(target_list) == [4, 6, 8]


def test_monad_laws():
    # 1. Left identity: bind(pure(x))(f) == f(x)
    safe_invert = lam(lambda x: Nothing if x == 0 else Some(1.0 / x))
    assert bind(pure_maybe(4))(safe_invert) == safe_invert(4) == Some(0.25)
    assert bind(pure_maybe(0))(safe_invert) == safe_invert(0) == Nothing

    # 2. Right identity: bind(m)(pure) == m
    m1 = Some("value")
    m2 = Nothing
    assert bind(m1)(pure_maybe) == m1
    assert bind(m2)(pure_maybe) == m2

    # 3. Associativity: bind(bind(m)(f))(g) == bind(m)(\x. bind(f(x))(g))
    safe_sqrt = lam(lambda x: Nothing if x < 0 else Some(x ** 0.5))
    m = Some(16.0)
    left = bind(bind(m)(safe_invert))(safe_sqrt)
    nested = lam(lambda x: bind(safe_invert(x))(safe_sqrt))
    right = bind(m)(nested)
    assert left == right == Some(0.25)


def test_symbolic_representation_and_church_reduction():
    # Verify pure symbolic properties of category terms
    # In pure lambda calculus, a pair (a, b) is \f. f a b
    pair_church = parse(r"\f. f a b")

    # Applying fst.term to Church pair reduces to 'a'
    fst_applied = App(fst.term, pair_church)
    assert fst_applied.normalize() == Var("a")

    # Applying snd.term to Church pair reduces to 'b'
    snd_applied = App(snd.term, pair_church)
    assert snd_applied.normalize() == Var("b")

    # Evaluation morphism: eval (pair f x) reduces to (f x)
    pair_fn_arg = parse(r"\k. k f x")
    eval_applied = App(eval_morphism.term, pair_fn_arg)
    assert eval_applied.normalize() == App(Var("f"), Var("x"))

    # Coproduct: inl and inr with fanin
    # fanin f g (inl x) reduces to (f x)
    church_inl_x = App(CHURCH_INL, Var("x"))
    fanin_term = App(App(App(CHURCH_FANIN, Var("f")), Var("g")), church_inl_x)
    assert fanin_term.normalize() == App(Var("f"), Var("x"))

    # fanin f g (inr y) reduces to (g y)
    church_inr_y = App(CHURCH_INR, Var("y"))
    fanin_term_right = App(App(App(CHURCH_FANIN, Var("f")), Var("g")), church_inr_y)
    assert fanin_term_right.normalize() == App(Var("g"), Var("y"))


def test_category_primitives_publish_their_symbolic_semantics():
    assert fanout.term == CHURCH_FANOUT
    assert bimap_product.term == CHURCH_BIMAP_PRODUCT
    assert curry.term == CHURCH_CURRY
    assert uncurry.term == CHURCH_UNCURRY
    assert bimap_coproduct.term == CHURCH_BIMAP_COPRODUCT
    assert pure_maybe.term == CHURCH_SOME
    assert kleisli_compose.term == KLEISLI_COMPOSE_TERM

    value = Var("value")
    assert inl(value).normalize() == parse(r"\l. \r. l value")
    assert inr(value).normalize() == parse(r"\l. \r. r value")

    paired = fanout(identity)(identity)
    assert App(paired.term, value).normalize() == parse(
        r"\pair. pair value value"
    )

    curried_fst = curry(fst)
    assert curried_fst.term.normalize() == parse(r"\x. \y. x")
    assert uncurry(curried_fst).term.normalize() == CHURCH_FST


def test_kleisli_category_laws():
    # Kleisli arrows: A -> Maybe[B]
    safe_div = lam(lambda x: Nothing if x == 0 else Some(12 / x))
    safe_sqrt = lam(lambda x: Nothing if x < 0 else Some(x ** 0.5))
    safe_sub_one = lam(lambda x: Some(x - 1))

    # 1. Left identity: pure >=> f == f
    id_left = kleisli_compose(pure_maybe)(safe_div)
    assert id_left(4) == safe_div(4) == Some(3.0)
    assert id_left(0) == safe_div(0) == Nothing

    # 2. Right identity: f >=> pure == f
    id_right = kleisli_compose(safe_div)(pure_maybe)
    assert id_right(4) == safe_div(4) == Some(3.0)
    assert id_right(0) == safe_div(0) == Nothing

    # 3. Associativity: (f >=> g) >=> h == f >=> (g >=> h)
    comp1 = kleisli_compose(kleisli_compose(safe_div)(safe_sub_one))(safe_sqrt)
    comp2 = kleisli_compose(safe_div)(kleisli_compose(safe_sub_one)(safe_sqrt))
    # 12 / 3 = 4, 4 - 1 = 3, sqrt(3)
    assert comp1(3) == comp2(3) == Some(3.0 ** 0.5)
    # 12 / 0 fails immediately
    assert comp1(0) == comp2(0) == Nothing


def test_natural_transformations_and_naturality_condition():
    double = lam(lambda x: x * 2)

    # 1. Naturality condition for alpha = maybe_to_list:
    # List(double) @ alpha == alpha @ Maybe(double)
    left_side = fmap(double) @ maybe_to_list
    right_side = maybe_to_list @ fmap(double)

    assert left_side(Some(21)) == right_side(Some(21)) == [42]
    assert left_side(Nothing) == right_side(Nothing) == []
    assert maybe_to_list.naturality_holds(double, Some(21))
    assert maybe_to_list.naturality_holds(double, Nothing)

    # 2. Vertical composition of Natural Transformations: (beta @ alpha)
    round_trip = list_to_maybe @ maybe_to_list
    assert round_trip(Some(100)) == Some(100)
    assert round_trip(Nothing) == Nothing
    assert round_trip.source is MAYBE_FUNCTOR
    assert round_trip.target is MAYBE_FUNCTOR

    # Endpoint metadata rejects an invalid vertical composition.
    with pytest.raises(TypeError):
        maybe_to_list @ maybe_to_list

    # 3. Either to Maybe natural transformation
    assert either_to_maybe(Right(42)) == Some(42)
    assert either_to_maybe(Left("err")) == Nothing


def test_comonads_env_and_store():
    # 1. Env (Coreader) Comonad: W[A] = (E, A)
    env_val = Env("config_path", 42)
    assert extract(env_val) == 42
    assert duplicate(env_val).extract().extract() == 42

    # extend with contextual query
    get_summary = lam(lambda e: f"{e.env}:{e.value}")
    extended_env = extend(get_summary)(env_val)
    assert extended_env.extract() == "config_path:42"

    # 2. Store (Costate) Comonad: W[A] = (S, S -> A)
    # Represents a 1D grid with a local accessor
    store = Store(5, lambda i: i * i)
    assert extract(store) == 25

    # Cobind / extend: local 3-point moving average
    local_avg = lam(
        lambda st: (
            st.accessor(st.pos - 1)
            + st.accessor(st.pos)
            + st.accessor(st.pos + 1)
        )
        / 3
    )
    smoothed = extend(local_avg)(store)
    # at pos=5: (16 + 25 + 36) / 3 = 77 / 3
    assert smoothed.extract() == 77 / 3


def test_recursion_schemes_cata_ana_hylo():
    # Peano Natural Numbers using Maybe as the pattern functor: F[X] = Maybe[X]
    # Algebra: F[int] -> int
    peano_algebra = lam(lambda m: 0 if m.is_nothing else m.value + 1)

    # Coalgebra: int -> F[int]
    peano_coalgebra = lam(lambda n: Nothing if n <= 0 else Some(n - 1))

    # Catamorphism & Anamorphism
    to_peano = ana(peano_coalgebra)
    from_peano = cata(peano_algebra)

    # 3 -> Peano Fix structure -> 3
    three_fix = to_peano(3)
    assert isinstance(three_fix, Fix)
    assert from_peano(three_fix) == 3

    # Hylomorphism: fold and unfold combined without intermediate Fix
    # List pattern functor F_A[X] = 1 + A x X
    class Nil:
        def fmap(self, f: Any) -> Any:
            return self

    class Cons:
        def __init__(self, head: int, tail: Any):
            self.head = head
            self.tail = tail

        def fmap(self, f: Any) -> Any:
            return Cons(self.head, f(self.tail))

    fact_coalg = lam(lambda n: Nil() if n <= 1 else Cons(n, n - 1))
    fact_alg = lam(lambda node: 1 if isinstance(node, Nil) else node.head * node.tail)

    # Direct hylo calculation for factorial:
    factorial = hylo(fact_alg)(fact_coalg)
    assert factorial(5) == 120
    assert factorial(1) == 1


def test_categorical_optics_lens():
    # Lens into a 2D Point
    point = {"x": 10, "y": 20}
    lens_x = Lens(lambda p: p["x"], lambda p, v: {**p, "x": v})
    # Lens laws:
    # 1. Get-Set: set(s, get(s)) == s
    assert lens_x.set(point)(lens_x.get(point)) == point

    # 2. Set-Get: get(set(s, a)) == a
    assert lens_x.get(lens_x.set(point)(99)) == 99

    # 3. Set-Set: set(set(s, a1), a2) == set(s, a2)
    assert lens_x.set(lens_x.set(point)(50))(100) == lens_x.set(point)(100)

    # Modify
    double = lam(lambda x: x * 2)
    modified = lens_x.modify(double)(point)
    assert modified == {"x": 20, "y": 20}

    # Lens Composition: nested structures
    player = {"info": {"health": 100, "mana": 50}}
    lens_info = Lens(lambda p: p["info"], lambda p, v: {**p, "info": v})
    lens_health = Lens(lambda i: i["health"], lambda i, v: {**i, "health": v})

    composed_lens = lens_health @ lens_info
    assert composed_lens.get(player) == 100
    healed = composed_lens.set(player)(150)
    assert healed == {"info": {"health": 150, "mana": 50}}


def test_fmap_requires_an_explicit_functor_protocol():
    class AccidentalMap:
        def map(self, function: Any) -> Any:
            return function(1)

    class Box:
        def __init__(self, value: Any):
            self.value = value

        def fmap(self, function: Any) -> Any:
            return Box(function(self.value))

    with pytest.raises(TypeError):
        fmap(identity)(AccidentalMap())

    mapped = fmap(lam(lambda value: value + 1))(Box(3))
    assert isinstance(mapped, Box)
    assert mapped.value == 4
    assert functor_for([1, 2]) is LIST_FUNCTOR
    assert functor_for(Some(1)) is MAYBE_FUNCTOR


def test_category_combinators_reject_invalid_inputs():
    with pytest.raises(TypeError):
        fanout(1)(identity)
    with pytest.raises(TypeError):
        uncurry(identity)
    with pytest.raises(TypeError):
        fanin(identity)(identity)("not Either")
    with pytest.raises(TypeError):
        extract(object())
    with pytest.raises(TypeError):
        bind(Some(1))(lam(lambda value: value + 1))
    with pytest.raises(TypeError):
        bind([1])(lam(lambda value: (value,)))
