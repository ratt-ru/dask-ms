# -*- coding: utf-8 -*-

import logging

import dask
import dask.array as da
import numpy as np
from dask.highlevelgraph import HighLevelGraph

from daskms.array_api_utils import is_array_api_obj, to_device_cpu
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.dataset import Dataset
from daskms.dataset_schema import DatasetSchema
from daskms.descriptors.builder import AbstractDescriptorBuilder
from daskms.descriptors.builder_factory import (
    filename_builder_factory,
    string_builder_factory,
)
from daskms.casa_table import (
    CasaTable,
    build_index,
    create_ms,
    create_table,
    ninstances,
)
from daskms.table import table_exists
from daskms.utils import table_path_split

log = logging.getLogger(__name__)


def putter_wrapper(data, rows, casa_table, column, block_info=None):
    """
    Write a block of ``column`` into the table.

    Called through :func:`dask.array.map_blocks`, whose ``block_info``
    supplies the block's half-open extent along each non-row dimension.
    ``rows`` holds the block's row ids, with singleton non-row dimensions.
    arcae writes rows in the order given, sorting them internally,
    so ``rows`` need not be sorted.

    Returns
    -------
    success : :class:`numpy.ndarray`
        singleton array containing True,
        having the same dimensionality as the input data.
    """
    info = block_info[0]
    shape = info["shape"]
    extents = info["array-location"][1:]
    rows = np.asarray(rows).ravel()

    # Only select along the non-row dimensions if they are chunked
    if all(tuple(extent) == (0, s) for extent, s in zip(extents, shape[1:])):
        extents = ()

    if isinstance(data, dict):
        # Dimension slicing is not supported for variably shaped data.
        if extents:
            raise ValueError(
                "Chunked writes for secondary dimensions "
                "unsupported for dictionary data"
            )

        # Variably shaped rows differ in shape from one another, so each
        # is written individually -- arcae can only address a ragged
        # column one row at a time. Key "r<i + 1>" holds rows[i].
        for i, row in enumerate(rows):
            value = np.asarray(data["r%d" % (i + 1)])
            # Restore the leading row dimension for scalar cells
            if value.ndim == 0:
                value = value[None]
            row = int(row)
            casa_table.instance.putcol(column, value, index=(slice(row, row + 1),))

    elif is_array_api_obj(data):
        data = np.asarray(to_device_cpu(data))

        # An empty block has nothing to write
        if len(rows) > 0:
            index = build_index(rows, extents)
            casa_table.instance.putcol(column, data, index=index)
    else:
        raise TypeError(f"data {type(data)} must be a numpy array or dict")

    return np.full((1,) * len(shape), True)


def descriptor_builder(table, descriptor):
    if descriptor is None:
        return filename_builder_factory(table)
    elif isinstance(descriptor, AbstractDescriptorBuilder):
        return descriptor
    else:
        return string_builder_factory(descriptor)


def _writable_table_proxy(table_name):
    return CasaTable.from_table(table_name, ninstances=ninstances(), readonly=False)


def _create_table(table_name, datasets, columns, descriptor):
    builder = descriptor_builder(table_name, descriptor)
    schemas = [DatasetSchema.from_dataset(ds, columns) for ds in datasets]
    table_desc, dminfo = builder.execute(schemas)

    root, table, subtable = table_path_split(table_name)
    table_path = root / table

    from daskms.descriptors.ms import MSDescriptorBuilder
    from daskms.descriptors.ms_subtable import MSSubTableDescriptorBuilder

    if not subtable and isinstance(builder, MSDescriptorBuilder):
        table_path = str(table_path)

        # Create the MS
        create_ms(table_path, table_desc=table_desc, dminfo=dminfo).close()

        return _writable_table_proxy(table_path)
    elif subtable:
        # NOTE(sjperkins)
        # Recreate the subtable path with OS separator components
        # This avoids accessing the subtable via the main table
        # (e.g. WSRT.MS::SOURCE)
        # which can cause lock issues as the subtables seemingly
        # inherit the parent table lock
        subtable_path = str(table_path / subtable)

        # Create the subtable
        if isinstance(builder, MSSubTableDescriptorBuilder):
            # arcae links a standard subtable into its parent's keyword
            # set as part of creating it
            create_ms(
                str(table_path),
                subtable=subtable,
                table_desc=table_desc,
                dminfo=dminfo,
            ).close()
        else:
            create_table(subtable_path, table_desc=table_desc, dminfo=dminfo).close()

            # A non-standard subtable must be registered against the
            # main table by hand
            table_proxy = _writable_table_proxy(str(table_path))
            table_proxy.instance.putkeywords({subtable: "Table: " + subtable_path})
            del table_proxy

        return _writable_table_proxy(subtable_path)
    else:
        # Create the table
        create_table(str(table_path), table_desc=table_desc, dminfo=dminfo).close()

        return _writable_table_proxy(str(table_path))


def _updated_table(table, datasets, columns, descriptor):
    table_proxy = _writable_table_proxy(table)
    table_columns = set(table_proxy.instance.columns())
    missing = set(columns) - table_columns

    # Add missing columns to the table
    if len(missing) > 0:
        # NOTE(sjperkins)
        # Updating a table with new columns with data managers
        # is a little tricky. Trying to update existing data managers
        # seems to incur casacore's internal wrath.
        #
        # Here, we
        # 1. Build a partial table description from missing variables
        # 2. Create Data Managers associated with missing variables,
        #    discarding any that currently exist on the table

        schemas = [DatasetSchema.from_dataset(ds, missing) for ds in datasets]
        builder = descriptor_builder(table, descriptor)
        variables = builder.dataset_variables(schemas)
        default_desc = builder.default_descriptor()
        table_desc = builder.descriptor(variables, default_desc)

        # Original Data Manager Groups
        odminfo = {g["NAME"] for g in table_proxy.instance.getdminfo().values()}
        SENTINEL = object()

        # NOTE(JSKenyon): Add columns one at a time - this avoids issues when
        # adding multiple columns with different managers.
        for col in missing:
            _table_desc = {col: table_desc[col]}
            _dminfo = builder.dminfo(_table_desc)
            in_odminfo = any(
                v.get("NAME", SENTINEL) in odminfo for v in _dminfo.values()
            )
            _dminfo = {} if in_odminfo else _dminfo
            table_proxy.instance.addcols(_table_desc, _dminfo)

    return table_proxy


def _add_row_wrapper(table, rows, checkrow=-1):
    startrow = table.nrow()

    if checkrow != -1 and startrow != checkrow:
        raise ValueError("Inconsistent starting row %d %d" % (startrow, checkrow))

    table.addrows(rows)

    return np.arange(startrow, startrow + rows, dtype=np.int64)


def add_row_orders(data, table_proxy, prev=None):
    """
    Adds rows to a table and returns associated row orderings.

    We want to be able to handle undefined row chunk sizes (np.nan) when
    writing dask arrays to a new table. This means that we may not know
    the starting row, or the number of rows in each chunk upfront.
    This poses further challenges when adding new rows to a table.

    This function addresses this by ingesting a chunk of data. From this
    the number of rows in the chunk can be determined and added to the table.
    The ids of the added rows are then returned as a result, which is passed
    as input to the ``add_row_orders`` call, operating on an adjacent chunk
    of row data.

    Parameters
    ----------
    data : :class:`numpy.ndarray` or dict
        If a numpy array the first dimension :code:`data.shape[0]`
        should contain the number of rows.
        If a dict, the number of rows is
        set to the length of the dict.
    table_proxy : :class:`daskms.casa_table.CasaTable`
        Table Proxy object
    prev : :class:`numpy.ndarray` or None
        Row ids added by the previous link in the chain.
        This argument serves two purposes:

        1. It is used to check the *starting row* of the current
           row ordering: rows should be added directly after the last
           row in ``prev``. The check is skipped if ``prev`` is empty.
        2. When this function is embedded in a dask graph, it establishes a
           dependency on the dask task which creates the previous rows.

        Defaults to ``None``, which indicates the first link in the chain.
        Rows are then added from the table's current row count.

    Returns
    -------
    :class:`numpy.ndarray`
        The ids of the added rows.
    """
    if is_array_api_obj(data):
        rows = data.shape[0]
    elif isinstance(data, dict):
        rows = len(data)
    else:
        raise TypeError(f"data {type(data)} must be a numpy array or dict")

    # NOTE(sjperkins)
    # arcae serialises writes onto a single instance, but reading the row
    # count and then adding rows is still check-then-act, so these calls
    # are chained sequentially by add_row_order_factory rather than run
    # concurrently.
    #
    # This is the first link in the chain,
    # or the previous link added no rows
    if prev is None or len(prev) == 0:
        return _add_row_wrapper(table_proxy.instance, rows, -1)
    else:
        # There's a previous link in the chain
        return _add_row_wrapper(table_proxy.instance, rows, int(prev[-1]) + 1)


def add_row_order_factory(table_proxy, datasets):
    """
    Generate arrays which add the appropriate rows for each array row chunk
    of a dataset, as well as returning the appropriate row ordering
    for that chunk

    Each array chunk (and by implication dataset) is linked by a call to
    :func:`daskms.writes.add_row_orders` to a previous chunk,
    either in the same array or the previous array.

    This establishes an order on how:

    1. Rows are added to the table.
    2. Column writes are performed.

    Returns
    -------
    list of :class:`dask.array.Array`
        row orderings for each dataset
    """
    prev_key = None
    prev_deps = []
    row_add_ops = []

    for di, ds in enumerate(datasets):
        data_vars = ds.data_vars
        found = False

        for k, v in data_vars.items():
            dims = v.dims
            array = v.data

            # Need something with a row dimension
            if not dims[0] == "row":
                continue

            found = True
            token = dask.base.tokenize(array)
            name = "-".join(("add-rows", str(di), token))
            layers = {}

            for b in range(array.numblocks[0]):
                key = (name, b)
                array_key = (array.name, b) + (0,) * (array.ndim - 1)
                layers[key] = (add_row_orders, array_key, table_proxy, prev_key)
                prev_key = key

            graph = HighLevelGraph.from_collections(name, layers, prev_deps + [array])
            chunks = (array.chunks[0],)
            row_adds = da.Array(graph, name, chunks, dtype=np.int64)
            row_add_ops.append(row_adds)
            prev_deps = [row_adds]

            break

        if not found:
            raise ValueError(
                "Couldn't find an array with "
                "which to establish a row ordering "
                "in dataset %d" % di
            )

    return row_add_ops


def _write_datasets(
    table, table_proxy, datasets, columns, descriptor, table_keywords, column_keywords
):
    _, table_name, subtable = table_path_split(table)
    table_name = "::".join((table_name, subtable)) if subtable else table_name
    row_orders = []

    # Put table and column keywords
    _put_keywords(table_proxy.instance, table_keywords, column_keywords)

    # Sort datasets on (not has "ROWID", index) such that
    # datasets with ROWID's are handled first, while
    # those without (which imply appends to the MS)
    # are handled last
    sorted_datasets = sorted(
        enumerate(datasets), key=lambda t: ("ROWID" not in t[1].data_vars, t[0])
    )

    # Establish row orders for each dataset
    for di, ds in sorted_datasets:
        try:
            rowid = ds.ROWID.data
        except AttributeError:
            # Add operation
            # No ROWID's, assume they're missing from the table
            # and remaining datasets. Generate addrows
            # NOTE(sjperkins)
            # This could be somewhat brittle, but exists to
            # update MS empty subtables once they've been
            # created along with the main MS by a call to default_ms.
            # Users could also it to append rows to an existing table.
            # An xds_append_to_table may be a better solution...
            last_datasets = datasets[di:]
            last_row_orders = add_row_order_factory(table_proxy, last_datasets)

            row_orders.extend(last_row_orders)
            # We have established row orders for all datasets
            # at this point, quit the loop
            break
        else:
            # Update operation, writing to the existing rows in ROWID
            row_orders.append(rowid)

    assert len(row_orders) == len(datasets)

    datasets = []

    for (di, ds), row_order in zip(sorted_datasets, row_orders):
        # Hold the variables representing array writes
        write_vars = {}

        # Generate a dask array for each column
        for column in columns:
            try:
                variable = ds.data_vars[column]
            except KeyError:
                log.warning("Ignoring '%s' not present on dataset %d" % (column, di))
                continue
            else:
                full_dims = variable.dims
                array = variable.data

            if not isinstance(array, da.Array):
                raise TypeError(
                    "%s on dataset %d is not a dask Array "
                    "but a %s" % (column, di, type(array))
                )

            if not (
                np.isnan(row_order.shape[0])
                if np.isnan(array.shape[0])
                else row_order.shape[0] == array.shape[0]
                and all(
                    np.isnan(a) if np.isnan(b) else a == b
                    for a, b in zip(row_order.chunks[0], array.chunks[0])
                )
            ):
                raise ValueError(
                    f"ROWID shape and/or chunking does not match that of {column}"
                )

            # Give the row ids singleton non-row dimensions,
            # so that map_blocks aligns them with the array's rows
            rows = row_order[(slice(None),) + (None,) * (array.ndim - 1)]

            # Name of the dask array representing this column
            token = dask.base.tokenize(di, row_order, table_proxy, column, array)
            name = "".join(("write~", column, "-", table_name, "-", token))

            # All dims shrink to 1, a single bool is returned
            write_col = da.map_blocks(
                putter_wrapper,
                array,
                rows,
                table_proxy,
                column,
                chunks=tuple((1,) * n for n in array.numblocks),
                dtype=bool,
                meta=np.empty((0,) * array.ndim, dtype=bool),
                name=name,
            )

            write_vars[column] = (full_dims, write_col)

        # Transfer any partition information over to the write dataset
        partition = ds.attrs.get(DASKMS_PARTITION_KEY, False)

        if not partition:
            attrs = None
        else:
            attrs = {
                DASKMS_PARTITION_KEY: partition,
                **{k: getattr(ds, k) for k, _ in partition},
            }

        # Append a dataset with the write operations
        datasets.append(Dataset(write_vars, attrs=attrs))

    # Return an empty dataset
    if len(datasets) == 0:
        return Dataset({})

    return datasets


DELKW = object()


def _put_keywords(table, table_keywords, column_keywords):
    if table_keywords is not None:
        for k, v in table_keywords.items():
            if v == DELKW:
                table.removekeyword(k)
            else:
                table.putkeywords({k: v})

    if column_keywords is not None:
        for column, keywords in column_keywords.items():
            for k, v in keywords.items():
                if v == DELKW:
                    table.removecolkeyword(column, k)
                else:
                    table.putcolkeywords(column, {k: v})

    return True


def write_datasets(
    table,
    datasets,
    columns,
    descriptor=None,
    table_keywords=None,
    column_keywords=None,
    table_proxy=False,
):
    # Promote datasets to list
    if isinstance(datasets, tuple):
        datasets = list(datasets)
    elif not isinstance(datasets, list):
        datasets = [datasets]

    # If ALL is requested
    if columns == "ALL":
        columns = list(set(ds.data_vars.keys()) for ds in datasets)

        if len(columns) > 0:
            columns = list(sorted(set.union(*columns)))

    if not table_exists(table):
        tp = _create_table(table, datasets, columns, descriptor)
    else:
        tp = _updated_table(table, datasets, columns, descriptor)

    write_datasets = _write_datasets(
        table,
        tp,
        datasets,
        columns,
        descriptor=descriptor,
        table_keywords=table_keywords,
        column_keywords=column_keywords,
    )

    if table_proxy:
        return write_datasets, tp

    return write_datasets
