# -*- coding: utf-8 -*-

"""casacure backend selection for dask-ms.

casacure (https://github.com/tmolteno/casacure) is a pure-Rust drop-in
replacement for the casacore table system, exposing ``casacure.tables`` with
the same surface as ``casacore.tables`` (plus the helper function set).

dask-ms talks to the backend exclusively through ``lazy_import("casacore.tables")``,
so selecting casacure is a matter of aliasing ``casacore`` -> ``casacure`` in
``sys.modules`` before any of those lazy imports are resolved.

Select it with the environment variable::

    DASK_MS_BACKEND=casacure python ...

With the variable set, casacure must be importable (``pip install casacure``,
or ``pip install "dask-ms[casacure]"``); otherwise an ``ImportError`` is
raised.  Without the variable, dask-ms keeps using the real python-casacore as
before (its hard dependency).
"""

import os
import sys

_BACKEND_MODULES = ("casacore", "casacore.tables")


def activate_casacure_backend() -> None:
    """Alias ``casacore`` -> ``casacure`` when ``DASK_MS_BACKEND=casacure``.

    Never overrides an already-imported ``casacore`` (e.g. a second import of
    dask-ms in a process where the backend was already resolved).
    """
    if os.environ.get("DASK_MS_BACKEND", "").lower() != "casacure":
        return
    if all(m in sys.modules for m in _BACKEND_MODULES):
        return
    try:
        import casacure
        import casacure.tables as tables
    except ImportError as exc:
        raise ImportError(
            "DASK_MS_BACKEND=casacure requires the 'casacure' package "
            "(pip install casacure)"
        ) from exc
    sys.modules.setdefault("casacore", casacure)
    sys.modules.setdefault("casacore.tables", tables)
