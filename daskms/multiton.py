"""dask-ms's use of the :class:`~rarg_python_patterns.Multiton` pattern.

The Multiton itself lives in ``rarg-python-patterns``; this module holds
the one piece of configuration dask-ms needs on top of it, and is the
single import site for the pattern within dask-ms.
"""

from __future__ import annotations

from typing import Any, Callable

from rarg_python_patterns import (
    FrozenKey,
    Multiton,
    freeze,
    normalise_args,
    register_freezer,
)

__all__ = [
    "FactoryFunctionT",
    "FrozenKey",
    "Multiton",
    "freeze",
    "normalise_args",
]


FactoryFunctionT = Callable[..., Any]


@register_freezer(Multiton)
def _freeze_multiton(arg: Multiton) -> FrozenKey:
    """Represent a multiton handle by its own key.

    Multitons are routinely passed as factory arguments to other
    multitons -- a TAQL query over a table, for instance. Freezing the
    handle to itself would put a strong reference to the parent in the
    child's cache key, keeping the parent alive for as long as the child
    is cached. The key alone identifies the parent just as precisely and
    holds nothing open.
    """
    return arg._key
