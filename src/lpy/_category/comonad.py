"""Protocol-based comonad operations."""

from __future__ import annotations

from typing import Any

from ..runtime import Lambda, lam
from .common import ensure_unary
from .protocols import (
    Morphism,
    SupportsDuplicate,
    SupportsExtend,
    SupportsExtract,
)


@lam
def extract(comonad: object) -> object:
    """Apply the comonad counit."""
    if isinstance(comonad, SupportsExtract):
        return comonad.extract()
    raise TypeError(f"extract does not support {type(comonad).__name__}")


@lam
def duplicate(comonad: object) -> object:
    """Duplicate a comonadic context."""
    if isinstance(comonad, SupportsDuplicate):
        return comonad.duplicate()
    raise TypeError(f"duplicate does not support {type(comonad).__name__}")


@lam
def extend(
    function: Morphism[Any, Any],
    comonad: object,
) -> object:
    """Map a context-aware function over a comonad."""
    function_lambda: Lambda[Any, Any] = ensure_unary(function, "extend function")
    if isinstance(comonad, SupportsExtend):
        return comonad.extend(function_lambda)
    raise TypeError(f"extend does not support {type(comonad).__name__}")
