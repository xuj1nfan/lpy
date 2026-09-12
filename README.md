# lpy
`lpy` 把 Python callable 变成严格柯里化的 `Lambda`，可以直接参与普通 Python 程序；每个对象同时保留一个基于 De Bruijn indices 的 Lambda 演算视图。

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
