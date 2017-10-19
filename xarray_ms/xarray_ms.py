import collections
from collections import OrderedDict
import logging
from functools import partial
import itertools
import sys

import six

try:
    from cytoolz import pluck, merge
except ImportError:
    from toolz import pluck, merge

import attr
import dask
import dask.array as da
from dask.array.core import getter
import numpy as np
import xarray as xr

from .known_table_schemas import registered_schemas

log = logging.getLogger("xarray-ms")

def _table_proxy(oargs, okwargs, attr, *args, **kwargs):
    """
    Proxies attribute access on a cached :class:`pyrap.tables.table` file.

    `oargs` and `okwargs` are arguments passed to
    the :class:`pyrap.tables.table` constructor. `oargs` can usually
    just be set to the name of the table as this is the only positional
    arg at the moment.

    Then `attr` is the attribute to access on the table
    while `args` and `kwargs` are the arguments pass to `attr` if
    it is `callable`.

    The table locking is set to `lockoptions='user'` and access to
    `attr` is guarded by calls to :meth:`pyrap.tables.table.lock`
    and :meth:`pyrap.tables.table.unlock`. Depending on the presence
    and value associated with the `readonly` key in `okwargs`,
    read or write locks will be used. For example `{'readonly':False }`
    will cause write locks to be requested.

    .. code-block:: python

        FLAGS = _table_proxy("WSRT.MS", {'readonly':True},
                    "getcol", "FLAGS", startrow=0, nrow=10)

    Parameters
    ----------
    oargs : string or tuple or list
        positional arguments to pass to pt.table constructor.
        Usually just the name of the table since everything else
        in :meth:`pyrap.tables.table` a kwarg.
    okwargs : dict
        keyword arguments to pass to pt.table constructor
    attr : string
        attribute to access or method to call on table
    *args (optional):
        Positional arguments passed to `attr` if `attr` is callable
    **kwargs (optional):
        Keyword arguments passed to `attr` if `attr` is callable
    """

    from file_cache import FILE_CACHE
    import pyrap.tables as pt

    # Convert oargs to tuple if singleton
    if not isinstance(oargs, (tuple, list)):
        oargs = (oargs,)

    # Should we request a write-lock?
    write_lock = okwargs.get('readonly', True) == False

    # Warn that we ignore lockoptions
    lockoptions = okwargs.pop('lockoptions', None)

    if lockoptions is not None and not lockoptions == 'user':
        log.warn("'lockoptions=%s' ignored by _table_proxy. "
                    "Locking is automatically handled in 'user' "
                    "mode" % lockoptions)

    # Get the table from the cache
    with FILE_CACHE.open(pt.table, lockoptions='auto', *oargs, **okwargs) as table:
        try:
            # Acquire a lock and get the attr
            table.lock(write=write_lock)
            fn_or_attr = getattr(table, attr)

            # If the attribute is callable, invoke it
            if callable(fn_or_attr):
                return fn_or_attr(*args, **kwargs)

            # Its just an attribute, return it
            return fn_or_attr

        finally:
            # Release the lock
            table.unlock()

# Map  column string types to numpy/python types
TABLE_TO_PY_TYPE = {
    'INT': np.int32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
    'BOOLEAN': np.bool,
    'COMPLEX': np.complex64,
    'DCOMPLEX': np.complex128,
    'STRING': object,
}

ColumnConfig = attr.make_class("ColumnConfig", ["column", "shape",
                                                "dims", "chunks", "dtype"])

def column_configuration(table, column, chunks=None, table_schema=None):
    """
    Constructs a :class:`ColumnConfig` for the given
    table, column and row chunks.

    Infers the following:

        (1) column shape
        (2) string dimensions
        (3) data type
        (4) chunks

    Shape is determined via the following three strategies,
    and in order of robustness:

        (1) If coldesc['option'] & 4 (FixedShape) is True, use coldesc['shape']
        (2) If coldesc['ndim'] is present use the shape of the
            first row of the column as the shape.
        (3) Otherwise assume the shape is (nrows,)

    Parameters
    ----------
    table : string
        CASA Table path
    column : string
        Table column
    chunks (optional) : int
        Number of chunks to use for each row
    table_schema (optional) : string or dict
        schema

    Returns
    -------
    :class:`ColumnConfig`
        Object describing the properties of the column,
        including `shape`, `chunks` and `dtype`.

    """
    okwargs = {'readonly':True}

    nrows = _table_proxy(table, okwargs, 'nrows')
    coldesc = _table_proxy(table, okwargs, 'getcoldesc', column)
    option = coldesc['option']

    if chunks is None:
        chunks = 10000

    if table_schema is None:
        table_schema = {}

    # FixedShape
    if option & 4:
        try:
            extra = tuple(coldesc['shape'].tolist())
        except KeyError:
            raise ValueError("'%s' column descriptor option '%d' specifies "
                            "a FixedShape but no 'shape' attribute was found "
                            "in the column desciptor" % (column, option))
        shape = (nrows,) + extra
        chunks = (chunks,) + extra
    # Variably shaped...
    else:
        # Perhaps we know something about the number of dimensions
        try:
            ndim = coldesc['ndim']
        except KeyError:
            # We have no idea what the shape is.
            # Set it to (nrows,), allegedly
            shape = (nrows,)
            chunks = (nrows,)

            log.warn("Inferring variable column '%s' shape as '%s'. "
                        "No 'shape' or 'ndim'." % (column, shape))
        else:
            # Guess data shape from first data row
            try:
                data = _table_proxy(table, okwargs, "getcol",
                                column, startrow=0, nrow=1)
            except BaseException as e:
                ve = ValueError("Couldn't determine shape of "
                                "column '%s' because '%s'" % (column, e))
                raise ve, None, sys.exc_info()[2]

            if not len(data.shape[1:]) == ndim:
                raise ValueError("column '%s' ndim = '%d' but first data row "
                                 "has shape '%s'" % (column, ndim, data.shape[1:]))

            shape = (nrows,) + data.shape[1:]
            chunks  = (chunks,) + data.shape[1:]

            log.warn("Inferring variable column '%s' shape as '%s' "
                        "from first data row." % (column, shape))

    # Extract valueType
    try:
        value_type = coldesc['valueType']
    except KeyError:
        raise ValueError("Cannot infer dtype for column '{}'. "
                        "Table Column Description is missing "
                        "valueType. Description is '{}'"
                            .format(column, coldesc))

    # Try conversion to numpy/python type
    try:
        dtype = TABLE_TO_PY_TYPE[value_type.upper()]
    except KeyError:
        raise ValueError("No known conversion from Table type '{}' "
                        "to python/numpy type.".format(value_type))


    # Create dimension strings
    ndim = len(shape)

    # Get a registered table schema
    if isinstance(table_schema, six.string_types):
        table_schema = registered_schemas().get(table_schema, {})

    # Attempt to get dimension schema for column
    if isinstance(table_schema, collections.Mapping):
        try:
            dims = ("rows", ) + table_schema[column].dims
        except KeyError:
            dims = ("rows",) + tuple('%s-dim%d' % (column, i) for i in range(1, ndim))
    else:
        raise TypeError("Invalid table_schema type '%s'" % (type(table_schema)))

    if not ndim == len(dims):
        raise ValueError("Length of dims '%s' does not match ndim '%d'" % (dims, ndim))

    chunks = da.core.normalize_chunks(chunks, shape=shape)
    return ColumnConfig(column, shape, dims, chunks, dtype)

def consecutive(index, stepsize=1):
    """ Partition index into list of arrays of consecutive indices """
    return np.split(index, np.where(np.diff(index) != stepsize)[0]+1)

def table_getcol_runs(table, col_cfg, runs):
    """
    Creates a dask array backed by calls to :meth:`pyrap.tables.table.getcol`
    on `runs` of consecutive indices.

    Parameters
    ----------
    table : string
        CASA Table path
    col_cfg : :class:`ColumnConfig`
        Column configuration
    runs : list of :class:`numpy.ndarray`
        List of consecutive index arrays

    Returns
    -------
    :class:`dask.array.Array`
        A dask array representing the given colun
    """

    # Reconfigure the row chunks using the supplied runs
    row_chunks = tuple(run[-1]-run[0]+1 for run in runs)
    chunks = (row_chunks,) + col_cfg.chunks[1:]
    assert sum(row_chunks) == col_cfg.shape[0]

    token = dask.base.tokenize(table, "getcol_runs", col_cfg.column, runs)
    name = '-'.join((table, "getcol_runs", col_cfg.column.lower(), token))

    # Given chunks == ((50, 50, 50, 50, 20), (3,))
    # The following results in
    # [((0, 50), (0, 3)),
    #  ((1, 50), (0, 3)),
    #  ((2, 50), (0, 3)),
    #  ...
    #  ((4, 20), (0, 3))]
    it = itertools.product(*(enumerate(dc) for dc in chunks))
    dsk = {}
    okwargs = {'readonly': True}

    # Create the dask graph for this array
    for product, run in zip(it, runs):
        key = (name,) + tuple(pluck(0, product))
        shape = tuple(pluck(1, product))

        dsk[key] = (partial(np.asarray, dtype=col_cfg.dtype),
                (partial(_table_proxy, startrow=run[0], nrow=len(run)),
                        table, okwargs, "getcol", col_cfg.column
                )
        )

    return da.Array(dsk, name, chunks, dtype=col_cfg.dtype)

def xds_to_table(dataset, data_arrays):
    """
    Constructs a dask array consisting of individual
    :meth:`pyrap.tables.table.putcol` calls
    on chunks of the `data_arrays` in the supplied `dataset`.

    .. code-block:: python

        pc = xds_to_table(ds, ["TIME", "ANTENNA1"])
        pc.compute()

    Parameters
    ----------
    dataset : :class:`xarray.Dataset`
        Dataset containing the `data_arrays`
    data_arrays : string or list
        List of data_array names to write out

    Returns
    -------
    :class:`dask.Array`
        A boolean dask array where each element is `True`
        and associated with
        a :meth:`pyrap.tables.table.putcol` command
        for each chunk in each array in `data_arrays`.
    """

    if not isinstance(data_arrays, (tuple, list)):
        data_arrays = [data_arrays]

    dsk = {}

    table = dataset.attrs['table']
    runs = dataset.attrs['runs']

    okwargs = {'readonly': False}
    success = np.ones(shape=(1,), dtype=np.bool)

    data_arrays = [(name, dataset[name.lower()].data)
                                for name in data_arrays]

    keys = []

    for array_name, array in data_arrays:
        # Reconfigure the row chunks using the supplied runs
        row_chunks = tuple(run[-1]-run[0]+1 for run in runs)
        chunks = (row_chunks,) + array.chunks[1:]
        assert sum(row_chunks) == array.shape[0]

        token = dask.base.tokenize(table, "putcol_runs", array_name, runs)
        name = '-'.join((table, "putcol_runs", name, token))

        # Given chunks == ((50, 50, 50, 50, 20), (3,))
        # The following results in
        # [((0, 50), (0, 3)),
        #  ((1, 50), (0, 3)),
        #  ((2, 50), (0, 3)),
        #  ...
        #  ((4, 20), (0, 3))]
        it = itertools.product(*(enumerate(dc) for dc in chunks))
        row_start = 0

        # Create the dask graph for this array
        for product, run in zip(it, runs):
            chunk_idx = tuple(pluck(0, product))
            key = (name,) + chunk_idx
            shape = list(pluck(1, product))
            row_end = row_start + shape[0]

            dsk[key] = (partial(_table_proxy, startrow=run[0], nrow=len(run)),
                                table, okwargs, "putcol", array_name,
                                (getter, array, slice(row_start, row_end)))

            row_start = row_end

            keys.append(key)

    # Construct the dask array internals
    array_names = tuple(tup[1].name for tup in data_arrays)
    token = dask.base.tokenize(table, *array_names)
    name = '-'.join((table, "putcol", token))
    chunks = da.core.normalize_chunks(1, shape=(len(keys),))
    dsk = merge(dsk, { (name, i) : k for i, k in enumerate(keys) })

    return da.Array(dsk, name, chunks, dtype=np.bool)


def _xds_from_table(table, chunks=None, runs=None, table_schema=None):
    """
    Creates an :class:`xarray.Dataset` backed by values in a CASA table

    Parameters
    ----------
    table : string
        CASA Table path
    chunks (optional): integer
        Row chunk size. Defaults to 10000.
    runs (optional): list of ndarrays
        List of arrays of consecutive indices
    table_schema (optional): dict or string
        schema

    Returns
    -------
    :class:`xarray.Dataset`
    """

    okwargs = { 'readonly': True }
    columns = _table_proxy(table, okwargs, "colnames")
    columns = sorted(columns)
    nrows = _table_proxy(table, okwargs, "nrows")

    def _gencfg(columns):
        for c in columns:
            try:
                yield c, column_configuration(table, c, chunks=chunks,
                                            table_schema=table_schema)
            except ValueError as e:
                log.warn("Ignoring column '%s'" % c, exc_info=True)


    row_range = np.arange(nrows)

    if chunks is None:
        chunks = 10000

    if runs is None:
        runs = [row_range]

    run_lengths = [len(r) for r in runs]

    if not sum(run_lengths) == len(row_range):
        raise ValueError("Sum of run lengths '%s' != number of rows '%s'" %
                            (run_lengths, len(row_range)))

    # Further subdivide any large runs into chunks
    runs = [a for run in runs
              for a in np.split(run, np.arange(chunks, run.size, chunks))]

    row_index = np.concatenate(runs)
    dataset_coords = { 'rows': row_range, 'msrows' : ('rows', row_index)}
    array_coords = { 'rows': row_range }

    make_da = lambda cfg: xr.DataArray(table_getcol_runs(table, cfg, runs),
                                        coords=array_coords, dims=cfg.dims)

    # Create an xarray dataset representing the table columns
    data_arrays = OrderedDict((c.lower(), make_da(cfg))
                                for c, cfg in _gencfg(columns))

    return xr.Dataset(data_arrays,
                    coords=dataset_coords,
                    attrs={'table': table, 'runs': runs})

def xds_from_table(table, chunks=None, table_schema=None):
    """
    Creates an :class:`xarray.Dataset` backed by values in a CASA table

    Parameters
    ----------
    table : string
        CASA Table path
    chunks (optional): integer
        Row chunk size. Defaults to 10000.
    table_schema (optional): dict or string
        schema

    Returns
    -------
    :class:`xarray.Dataset`
    """

    return _xds_from_table(table, chunks=None, table_schema=table_schema)

def xds_from_ms(ms, chunks=None, time_ordered=True):
    """
    Creates an :class:`xarray.Dataset` backed by values in a CASA Measurement Set.

    Parameters
    ----------
    ms : string
        CASA Measurement Set path.
    chunks (optional) : integer
        Row chunk size. Defaults to `10000`.
    time_ordered (optional) : bool
        If True, each `xarray.DataArray` on the Dataset
        will be ordered by the time dimension.
        Additionally the following coordinates will be created on the Dataset:

            1. `time_unique` containing the ordered unique timestamps.
            2. `time_chunks` containing the frequency of each unique timestamp.
            3. `time_offsets` containing the row index the start of each unique timestamp.

        Defaults to `True`.

    Returns
    -------
    :class:`xarray.Dataset`
    """

    if chunks is None:
        chunks = 100000

    okwargs = { 'readonly': True }
    nrows = _table_proxy(ms, okwargs, "nrows")
    row_range = np.arange(nrows)

    # If time ordering is requested, we read in the time column
    # and construct an index with timestamps as coordinates.
    # The indices grouped with each timestamp are stacked
    # in time ascending order to get a MS row index to nrows coordinate
    if time_ordered:
        time_cfg = column_configuration(ms, "TIME", chunks=chunks,
                                                table_schema="MS")

        # Subdivide row_range into runs of size `chunks`
        runs = np.split(row_range, np.arange(chunks, row_range.size, chunks))
        # Construct a dask array and reify it to a numpy array immediately
        times = table_getcol_runs(ms, time_cfg, runs).compute()
        # Construct a DataArray of time indices, with the actual
        # times as coordinates, but mediated through the 'aux'
        # coordinate so that xarray handles duplicate times correctly.
        # See https://stackoverflow.com/a/38073919/1611416
        time_index = xr.DataArray(np.arange(times.shape[0]),
                                    coords={ 'aux': ('time', times) },
                                    dims=["time"])


        # Construct a row_index such that the row indices associated
        # with each unique time are ordered consecutively.
        # Also identify unique times and their frequencies
        row_index = []
        unique_times = []
        time_chunks = [0]

        for ti, (t, dary) in enumerate(time_index.groupby('aux')):
            row_index.append(dary)
            time_chunks.append(len(dary))
            unique_times.append(t)

        row_index = np.concatenate(row_index)

        extra_arrays = {
            "time_unique" : (("utime",), unique_times),
            "time_chunks": (("utime",), np.asarray(time_chunks[1:])),
            "time_offsets" : (("utime",), np.cumsum(time_chunks[:-1])) }

        extra_coords = { "utime": np.arange(len(unique_times)) }

    else:
        row_index = row_range
        extra_arrays = {}
        extra_coords = {}

    # Compute consecutive runs of row indices
    runs = consecutive(row_index)

    # Create the Dataset and add extra coordinates
    xds = _xds_from_table(ms, chunks=chunks, runs=runs,
                                    table_schema="MS")

    # Create coordinates for channel, correlation
    # and polarisation dimensions
    extra_coords.update({ "chans": np.arange(xds.dims["chans"]),
                          "corrs": np.arange(xds.dims["corrs"]),
                          "pols": np.arange(xds.dims["corrs"])})

    # Add extra arrays and coordinates
    xds.update(extra_arrays)
    xds.coords.update(extra_coords)
    return xds
