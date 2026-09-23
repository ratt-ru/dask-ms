"""dask-ms's use of the :class:`~rarg_python_patterns.Multiton` pattern.

The Multiton itself lives in ``rarg-python-patterns``; this module is the
single import site for the pattern within dask-ms, so that a future move
to a different implementation touches one file.
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
    "register_freezer",
]


FactoryFunctionT = Callable[..., Any]
