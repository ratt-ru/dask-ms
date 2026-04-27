"""
Minimal vendored helpers from array-api-compat for array library detection
and CPU transfer. Adapted from:
https://github.com/data-apis/array-api-compat/blob/main/array_api_compat/common/_helpers.py

MIT License

Copyright (c) 2022 Consortium for Python Data API Standards

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
from functools import lru_cache


@lru_cache(100)
def _issubclass_fast(cls, modname, clsname):
    try:
        mod = sys.modules[modname]
    except KeyError:
        return False
    return issubclass(cls, getattr(mod, clsname))


def is_array_api_obj(x):
    """Return True if x is an array API compatible object."""
    cls = type(x)
    return (
        hasattr(x, "__array_namespace__")
        or _issubclass_fast(cls, "numpy", "ndarray")
        or _issubclass_fast(cls, "cupy", "ndarray")
        or _issubclass_fast(cls, "torch", "Tensor")
        or _issubclass_fast(cls, "dask.array", "Array")
        or _issubclass_fast(cls, "jax", "Array")
        or _issubclass_fast(cls, "sparse", "SparseArray")
    )


def to_device_cpu(x):
    """Move an array to CPU, returning the result."""
    cls = type(x)
    if _issubclass_fast(cls, "cupy", "ndarray"):
        return x.get()
    if _issubclass_fast(cls, "torch", "Tensor"):
        return x.to("cpu")
    return x
