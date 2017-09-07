import collections
import logging
from functools import partial
import itertools
from operator import getitem
import sys

import six

try:
    from cytoolz import pluck, merge
except ImportError:
    from toolz import pluck, merge

import attr
import dask.array as da
import dask.core
import numpy as np
import xarray as xr

from .known_table_schemas import registered_schemas

log = logging.getLogger("xarray-ms")

def _ms_proxy(oargs, okwargs, attr, *args, **kwargs):
    """
    Proxies attribute access on a cached :class:`pyrap.tables.table` file.

    `oargs` and `okwargs` are arguments passed to
    the :class:`pyrap.tables.table` constructor. `oargs` can usually
    just be set to the name of the MS as this is the only positional
    arg at the moment.

    Then `attr` is the attribute to access on the Measurement Set
    while `args` and `kwargs` are the arguments pass to `attr` if
    it is `callable`.

    The table locking is set to `lockoptions='user'` and access to
    `attr` is guarded by calls to :meth:`pyrap.tables.table.lock`
    and :meth:`pyrap.tables.table.unlock`. Depending on the presence
    and value associated with the `readonly` key in `okwargs`,
    read or write locks will be used. For example `{'readonly':False }`
    will cause write locks to be requested.

    .. code-block:: python

        FLAGS = _ms_proxy("WSRT.MS", {'readonly':True},
                    "getcol", "FLAGS", startrow=0, nrow=10)

    Parameters
    ----------
    oargs : string or tuple or list
        positional arguments to pass to pt.table constructor.
        Usually just the name of the ms since everything else
        is just a kwarg
    okwargs : dict
        keyword arguments to pass to pt.table constructor
    attr : string
        attribute to access or method to call on Measurement Set
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
        log.warn("'lockoptions=%s' ignored by _ms_proxy. "
                    "Locking is automatically handled in 'user' "
                    "mode" % lockoptions)

    # Get the table from the cache
    with FILE_CACHE.open(pt.table, lockoptions='auto', *oargs, **okwargs) as ms:
        try:
            # Acquire a lock and get the attr
            ms.lock(write=write_lock)
            fn_or_attr = getattr(ms, attr)

            # If the attribute is callable, invoke it
            if callable(fn_or_attr):
                return fn_or_attr(*args, **kwargs)

            # Its just an attribute, return it
            return fn_or_attr

        finally:
            # Release the lock
            ms.unlock()

# Map MS column string types to numpy/python types
MS_TO_NP_TYPE_MAP = {
    'INT': np.int32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
    'BOOLEAN': np.bool,
    'COMPLEX': np.complex64,
    'DCOMPLEX': np.complex128,
    'STRING': str,
}

ColumnConfig = attr.make_class("ColumnConfig", ["column", "shape",
                                                "dims", "chunks", "dtype"])

def column_configuration(ms, column, chunks=None, table_schema=None):
    """
    Constructs a :class:`ColumnConfig` for the given
    Measurement Set, column and row chunks.

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
    ms : string
        Measurement Set
    column : string
        Measurement Set column
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

    nrows = _ms_proxy(ms, okwargs, 'nrows')
    coldesc = _ms_proxy(ms, okwargs, 'getcoldesc', column)
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
                data = _ms_proxy(ms, okwargs, "getcol",
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
                        "MS Column Description is missing "
                        "valueType. Description is '{}'"
                            .format(column, coldesc))

    # Try conversion to numpy/python type
    try:
        dtype = MS_TO_NP_TYPE_MAP[value_type.upper()]
    except KeyError:
        raise ValueError("No known conversion from MS type '{}' "
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

    chunks = da.core.normalize_chunks(chunks, shape=shape)
    return ColumnConfig(column, shape, dims, chunks, dtype)

def consecutive(index, stepsize=1):
    """ Partition index into list of arrays of consecutive indices """
    return np.split(index, np.where(np.diff(index) != stepsize)[0]+1)

def ms_getcol_runs(ms, col_cfg, runs):
    """
    Creates a dask array backed by calls to :meth:`pyrap.tables.table.getcol`
    on `runs` of consecutive indices.

    Parameters
    ----------
    ms : string
        Measurement Set
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

    token = dask.base.tokenize(ms, "getcol_runs", col_cfg.column, runs)
    name = '-'.join((ms, "getcol_runs", col_cfg.column.lower(), token))

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

        dsk[key] = (partial(_ms_proxy, startrow=run[0], nrow=len(run)),
                            ms, okwargs, "getcol", col_cfg.column)

    return da.Array(dsk, name, chunks, dtype=col_cfg.dtype)

def xds_to_table(dataset, data_arrays):
    """
    Constructs a delayed call evaluating a dask
    graph that calls :meth:`pyrap.tables.table.putcol`
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
    :class:`dask.Delayed`
        delayed object
    """

    if not isinstance(data_arrays, (tuple, list)):
        data_arrays = [data_arrays]

    dsk = {}

    ms = dataset.attrs['ms']
    runs = dataset.attrs['runs']

    okwargs = {'readonly': False}

    data_arrays = [(name, dataset[name.lower()].data)
                                for name in data_arrays]

    for array_name, array in data_arrays:
        # Reconfigure the row chunks using the supplied runs
        row_chunks = tuple(run[-1]-run[0]+1 for run in runs)
        chunks = (row_chunks,) + array.chunks[1:]
        assert sum(row_chunks) == array.shape[0]

        token = dask.base.tokenize(ms, "putcol_runs", array_name, runs)
        name = '-'.join((ms, "putcol_runs", name, token))

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

            dsk[key] = (partial(_ms_proxy, startrow=run[0], nrow=len(run)),
                                ms, okwargs, "putcol", array_name,
                                (getitem, array, slice(row_start, row_end)))

            row_start = row_end

    # Return MS delayed evaluation of the putcol graph
    return dask.delayed(dask.get)(merge(dsk, *(t[1].dask for t in data_arrays)), dsk.keys())

def xds_from_table(ms, chunks=None, time_ordered=True, table_schema=None):
    """
    Creates an :class:`xarray.Dataset` backed by values in a CASA table

    Parameters
    ----------
    ms : string
        CASA Table path
    chunks (optional): integer
        Row chunk size. Defaults to 10000.
    time_ordered (optional): bool
        If True, the resulting arrays will be ordered
        by the time dimension. Defaults to True
    table_schema (optional): dict or string
        schema

    Returns
    -------
    :class:`xarray.Dataset`
    """
    okwargs = { 'readonly': True }
    columns = _ms_proxy(ms, okwargs, "colnames")

    if chunks is None:
        chunks = 10000

    def _gencfg(columns):
        for c in columns:
            try:
                yield c, column_configuration(ms, c, chunks=chunks,
                                            table_schema=table_schema)
            except ValueError as e:
                log.warn("Ignoring column '%s'" % c, exc_info=True)

    time_cfg = column_configuration(ms, "TIME", chunks=chunks,
                                            table_schema=table_schema)
    row_range = np.arange(time_cfg.shape[0])

    # If time ordering is requested, we read in the time column
    # and construct an index with timestamps as coordinates.
    # The indices grouped with each timestamp are stacked
    # in time ascending order to get a MS row index to nrows coordinate
    if time_ordered:
        # Subdivide row_range into runs of size `chunks`
        runs = np.split(row_range, np.arange(chunks, row_range.size, chunks))
        times = ms_getcol_runs(ms, time_cfg, runs).compute()
        # Construct a DataArray of time indices, with the actual
        # times as coordinates, but mediated through the 'aux'
        # coordinate so that xarray handles duplicate times correctly.
        # See https://stackoverflow.com/a/38073919/1611416
        time_index = xr.DataArray(np.arange(times.shape[0]),
                                    #coords= { 'time': times },
                                    coords={ 'aux': ('time', times) },
                                    dims=["time"])

        # Now concatenate the indices associated with each time
        # in ascending time order
        row_index = np.concatenate([dary.values for t, dary
                            in time_index.groupby('aux')])
    else:
        row_index = row_range

    # Compute consecutive runs of row indices
    runs = consecutive(row_index)

    # Further subdivide any large runs into chunks
    runs = [a for run in runs
              for a in np.split(run, np.arange(chunks, run.size, chunks))]

    dataset_coords = { 'rows': row_range, 'msrows' : ('rows', row_index)}
    array_coords = { 'rows': row_range }

    # Create an xarray dataset representing the Measurement Set columns
    data_arrays = { c.lower(): xr.DataArray(ms_getcol_runs(ms, cfg, runs),
                                            coords=array_coords,
                                            dims=cfg.dims)
                                for c, cfg in _gencfg(columns) }

    return xr.Dataset(data_arrays,
                    coords=dataset_coords,
                    attrs={'ms':ms, 'runs': runs})
