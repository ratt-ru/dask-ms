# -*- coding: utf-8 -*-

"""casacure, the table backend of dask-ms.

casacure (https://github.com/tmolteno/casacure) is a pure-Rust
implementation of the CASA table system, exposing ``casacure.tables`` with
the same surface as ``casacore.tables`` (plus the helper function set).
It is a core dependency of dask-ms-2 and is always the table backend: there
is nothing to select or configure at runtime.

dask-ms talks to the table system exclusively through
``lazy_import("casacore.tables")``, so activating casacure is a matter of
aliasing ``casacore`` -> ``casacure`` in ``sys.modules`` before any of those
lazy imports are resolved. That happens automatically when ``daskms`` is
imported.
"""

import sys


def activate_casacure_backend() -> None:
    """Alias ``casacore`` -> ``casacure``, unconditionally.

    Called on ``import daskms``. The alias is (re)installed on every call so
    that the table backend is always casacure, even when ``casacore`` was
    imported earlier in the process.
    """
    try:
        import casacure
        import casacure.tables as tables
    except ImportError as exc:
        raise ImportError(
            "dask-ms-2 requires the 'casacure' package, its CASA table "
            "backend (pip install casacure)"
        ) from exc

    sys.modules["casacore"] = casacure
    sys.modules["casacore.tables"] = tables
