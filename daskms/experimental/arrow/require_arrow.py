from functools import partial

from daskms.utils import requires

requires_arrow = partial(
    requires,
    "pip install dask-ms[arrow] for arrow support")
