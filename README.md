# lpy
`lpy` 是一个数学编程实验库：它把 Python callable 变成严格柯里化的 `Lambda`，保留基于 De Bruijn indices 的 Lambda 演算视图，并在同一运行时上提供积、余积、函子、单子、余单子、递归图式和 Lens 等范畴论抽象。

## Python 中直接使用

```python
from lpy import lam

identity = lam(lambda x: x)
increment = lam(lambda x: x + 1)

assert identity("hello") == "hello"
assert list(map(increment, [1, 2, 3])) == [2, 3, 4]

@lam
def add(x, y):
    return x + y

# 严格一元柯里化：每次只能传一个参数
assert add(1)(2) == 3
```

`lam` 接受任意 Python 函数体，但签名必须由一个或多个固定位置参数组成，不能有默认值、关键字专用参数或变长参数。函数只会在最后一个参数到齐时执行。

`Lambda[A, B]` 保留一元态射的输入和输出类型；多参数函数在类型层表示为嵌套的 `Lambda`，例如二元整数加法是 `Lambda[int, Lambda[int, int]]`。

## 函数组合

`f @ g` 表示数学上的 `f(g(x))`：

```python
double = lam(lambda x: x * 2)
increment_after_double = increment @ double

assert increment_after_double(3) == 7
```

组合的两侧必须都是当前只剩一个参数的 `Lambda`。

## 演算视图

```python
from lpy import App, SymbolicKind, Var, parse

identity = lam(lambda x: x)
assert identity.term == parse(r"\x. x")
assert identity.symbolic_kind is SymbolicKind.PURE

# Term 参数进行符号应用，不执行 Python
application = identity(Var("y"))
assert application == App(identity.term, Var("y"))
assert application.normalize() == Var("y")
```

静态分析能够表示纯函数应用以及常量、运算符、比较、属性和索引等常见表达式。复杂控制流或无法读取源码的 callable 会安全降级为不可约 Host 节点：

```python
assert increment.symbolic_kind in {
    SymbolicKind.EXTENDED,
    SymbolicKind.OPAQUE,  # 例如交互式环境无法取得源码
}
print(increment.symbolic_reason)
```

包装过程中不会执行用户函数。Host 节点在 `.term.normalize()` 中也永远不会执行；真实 Python 执行只通过 `Lambda` 调用发生。

如果自动分析不能准确表达一个函数，可以显式提供演算项：

```python
@lam(term=parse(r"\x. \y. x"))
def first(x, y):
    return x

assert first("a")("b") == "a"
```

## 纯 Lambda 演算

底层仍可独立使用：

```python
from lpy import Strategy

term = parse(r"(\x. x) y")
assert str(term.normalize()) == "y"
assert parse(r"\x. x") == parse(r"\z. z")  # α-等价

for stage in term.trace(Strategy.NORMAL_ORDER, max_steps=10):
    print(stage)
```

`Term` 对象只接受其他 `Term`/`Lambda` 做符号应用，不直接执行普通 Python 值。解析语法使用 ASCII 反斜线，例如 `r"\x y. x y"`。

## 范畴论（Category Theory）支持

`lpy` 提供了范畴论与笛卡尔闭范畴（CCC）的标准原语，兼具 Python 运行时执行与底层 Church 编码演算视图：

```python
from lpy import (
    identity, fst, snd, fanout, bimap_product,
    curry, uncurry, eval_morphism,
    Either, Left, Right, inl, inr, fanin,
    Maybe, Some, Nothing, fmap, bind, pure_maybe, kleisli_compose,
)

# 1. 恒等态射与单位元律 (id @ f == f, f @ id == f)
assert (identity @ increment)(5) == 6

# 2. 笛卡尔积：投影与配对态射 <f, g>(x) = (f(x), g(x))
paired = fanout(increment)(double)
assert paired(3) == (4, 6)
assert (fst @ paired)(3) == 4

# 3. 柯里化同构：Hom(A x B, C) ≅ Hom(A, C^B)
add_pair = lam(lambda p: p[0] + p[1])
curried_add = curry(add_pair)
assert curried_add(3)(4) == 7
assert uncurry(curried_add)((3, 4)) == 7

# 4. 余积（和类型）：Either 与 [f, g] 余配对
cases = fanin(double)(lam(lambda s: s.upper()))
assert cases(inl(10)) == 20
assert cases(inr("cat")) == "CAT"

# 5. 函子 (fmap) 与 单子 (bind)
assert fmap(double)(Some(5)) == Some(10)
assert fmap(double)(Nothing) == Nothing

safe_div = lam(lambda x: Nothing if x == 0 else Some(10 / x))
assert bind(Some(2))(safe_div) == Some(5.0)
assert bind(Some(0))(safe_div) == Nothing

# 6. Kleisli 箭头复合 (f >=> g)
safe_sqrt = lam(lambda x: Nothing if x < 0 else Some(x ** 0.5))
div_and_sqrt = kleisli_compose(safe_div)(safe_sqrt)
assert div_and_sqrt(2.5) == Some(2.0)  # sqrt(10 / 2.5) = sqrt(4) = 2.0
assert div_and_sqrt(0) == Nothing

# 7. 自然变换 (Natural Transformation) 与垂直复合
from lpy import maybe_to_list, list_to_maybe
assert (fmap(double) @ maybe_to_list)(Some(21)) == [42]
assert (list_to_maybe @ maybe_to_list)(Some(99)) == Some(99)

# 8. 余单子 (Comonads): Store & Env
from lpy import Store, extract, extend
store = Store(5, lambda i: i * i)
assert extract(store) == 25
local_average = lam(
    lambda s: (
        s.accessor(s.pos - 1)
        + s.accessor(s.pos)
        + s.accessor(s.pos + 1)
    ) / 3
)
smoothed = extend(local_average)(store)
assert smoothed.extract() == 77 / 3

# 9. 递归图式与 F-代数: cata, ana, hylo
from lpy import Fix, cata, ana, hylo
# Peano: F[X] = Maybe[X]
to_peano = ana(lam(lambda n: Nothing if n <= 0 else Some(n - 1)))
from_peano = cata(lam(lambda m: 0 if m.is_nothing else m.value + 1))
assert from_peano(to_peano(3)) == 3

# 10. 范畴论透镜 (Lens)
from lpy import Lens
lens_health = Lens(lambda p: p["health"], lambda p, v: {**p, "health": v})
player = {"health": 100, "name": "Hero"}
assert lens_health.get(player) == 100
assert lens_health.modify(double)(player) == {"health": 200, "name": "Hero"}
```

### 语义边界

- 运行时的积和余积分别使用 Python `tuple` 和 `Either`；`CHURCH_*` 常量是对应的纯符号表示，两者不会自动转换。
- `fmap` 通过 `FunctorAdapter` 选择 `Maybe`、`Either`、`Env`、`Store`、`list` 或 `tuple` 的映射规则；自定义容器需显式实现 `SupportsFmap` 的 `.fmap()`。
- `bind`、`extract`、`duplicate` 和 `extend` 也使用显式协议。内置 Monad 会检查 `bind` 返回的容器家族；自定义实现负责保证相应定律。
- `NaturalTransformation` 持有真正的源、目标 `FunctorAdapter` 对象，不是字符串标签。它会拒绝端点不匹配的垂直复合；`.naturality_holds(f, value)` 只检查一个有限样本，不是形式证明。
- `cata`、`ana` 和 `hylo` 目前使用 Python 递归，适合有限且深度适中的结构，不保证栈安全。

### 符号支持

`identity`、积/余积组合子、`curry`、`uncurry`、`eval_morphism`、`inl`、`inr` 和 `pure_maybe` 都有明确的纯 Church 项。`kleisli_compose` 使用含自由变量 `bind` 的抽象符号项。`fmap`、`bind`、余单子、递归图式和 Lens 依赖 Python 容器协议，因此其 `.term` 会如实标记为扩展或不透明表示。高级常量、adapter 和协议可从 `lpy.category` 导入。

## 开发

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy
pytest
python -m build
```

Hypothesis 性质测试覆盖 Functor、Monad、Comonad、Lens、curry/uncurry 和自然性定律；CI 在 Python 3.10–3.14 上运行全部检查。
