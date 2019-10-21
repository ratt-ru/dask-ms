# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import pyrap.tables as pt

from daskms.columns import dim_extents_array
from daskms.descriptors.builder import AbstractDescriptorBuilder
from daskms.descriptors.builder_factory import filename_builder_factory
from daskms.descriptors.builder_factory import string_builder_factory
from daskms.ordering import row_run_factory
from daskms.table import table_exists
from daskms.table_executor import executor_key
from daskms.table_proxy import TableProxy, WRITELOCK
from daskms.utils import table_path_split


log = logging.getLogger(__name__)


def ndarray_putcol(row_runs, table_future, column, data):
    """ Put data into the table """
    table = table_future.result()
    putcol = table.putcol
    rr = 0

    table.lock(write=True)

    try:
        for rs, rl in row_runs:
            putcol(column, data[rr:rr + rl], startrow=rs, nrow=rl)
            rr += rl

        table.flush()

    finally:
        table.unlock()


def multdim_str_putcol(row_runs, table_future, column, data):
    """ Put multidimensional string data into the table """
    table = table_future.result()
    putcol = table.putcol

    rr = 0

    table.lock(write=True)

    try:
        for rs, rl in row_runs:
            # Construct a dict with the shape and a flattened list
            chunk = data[rr:rr + rl]
            chunk = {'shape': chunk.shape, 'array': chunk.ravel().tolist()}
            putcol(column, chunk, startrow=rs, nrow=rl)
            rr += rl

        table.flush()

    finally:
        table.unlock()


def ndarray_putcolslice(row_runs, blc, trc, table_future, column, data):
    """ Put data into the table """
    table = table_future.result()
    putcolslice = table.putcolslice
    rr = 0

    table.lock(write=True)

    try:
        for rs, rl in row_runs:
            putcolslice(column, data[rr:rr + rl], blc, trc,
                        startrow=rs, nrow=rl)
            rr += rl

        table.flush()

    finally:
        table.unlock()


def multdim_str_putcolslice(row_runs, blc, trc, table_future, column, data):
    """ Put multidimensional string data into the table """
    table = table_future.result()
    putcol = table.putcol
    rr = 0

    table.lock(write=True)

    try:
        for rs, rl in row_runs:
            # Construct a dict with the shape and a flattened list
            chunk = data[rr:rr + rl]
            chunk = {'shape': chunk.shape, 'array': chunk.ravel().tolist()}
            putcol(column, chunk, blc, trc, startrow=rs, nrow=rl)
            rr += rl

        table.flush()

    finally:
        table.unlock()


def putter_wrapper(row_orders, *args):
    """
    Wrapper which should run I/O operations within
    the table_proxy's associated executor
    """
    # Infer number of shape arguments
    nextent_args = len(args) - 3
    # Extract other arguments
    table_proxy, column, data = args[nextent_args:]

    # Handle dask compute_meta gracefully
    if len(row_orders) == 0:
        return np.empty((0,)*len(data.shape), dtype=np.bool)

    row_runs, resort = row_orders

    if resort is not None:
        data = data[resort]

    # Infer output shape before possible list conversion
    out_shape = (1,) * len(data.shape)

    # NOTE(sjperkins)
    # python-casacore wants to put lists of objects, but
    # because dask.array handles ndarrays we're passed
    # ndarrays of python objects (strings).
    # Without this conversion python-casacore can segfault
    # See https://github.com/ska-sa/dask-ms/issues/42
    multidim_str = False

    if data.dtype == np.object:
        # Multi-dimensional strings, we need to pass dicts through
        if data.ndim > 1:
            multidim_str = True
        # We can just pass dictionaries through
        else:
            data = data.tolist()

    # There are other dimensions beside row
    if nextent_args > 0:
        blc, trc = zip(*args[:nextent_args])
        fn = multdim_str_putcolslice if multidim_str else ndarray_putcolslice
        table_proxy._ex.submit(fn, row_runs, blc, trc,
                               table_proxy._table_future,
                               column, data).result()
    else:
        fn = multdim_str_putcol if multidim_str else ndarray_putcol
        table_proxy._ex.submit(fn, row_runs,
                               table_proxy._table_future,
                               column, data).result()

    return np.full(out_shape, True)


def descriptor_builder(table, descriptor):
    if descriptor is None:
        return filename_builder_factory(table)
    elif isinstance(descriptor, AbstractDescriptorBuilder):
        return descriptor
    else:
        return string_builder_factory(descriptor)


def _writable_table_proxy(table_name):
    return TableProxy(pt.table, table_name, ack=False,
                      readonly=False, lockoptions='user',
                      __executor_key__=executor_key(table_name))


def _create_table(table_name, datasets, columns, descriptor):
    builder = descriptor_builder(table_name, descriptor)
    table_desc, dminfo = builder.execute(datasets)

    root, table, subtable = table_path_split(table_name)
    table_path = root / table

    from daskms.descriptors.ms import MSDescriptorBuilder
    from daskms.descriptors.ms_subtable import MSSubTableDescriptorBuilder

    if not subtable and isinstance(builder, MSDescriptorBuilder):
        table_path = str(table_path)

        # Create the MS
        with pt.default_ms(table_path, tabdesc=table_desc, dminfo=dminfo):
            pass

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
            with pt.default_ms_subtable(subtable, subtable_path,
                                        tabdesc=table_desc, dminfo=dminfo):
                pass
        else:
            with pt.table(subtable_path, table_desc, dminfo=dminfo, ack=False):
                pass

        # Add subtable to the main table
        table_proxy = _writable_table_proxy(str(table_path))
        table_proxy.putkeywords({subtable: "Table: " + subtable_path}).result()
        del table_proxy

        # Return TableProxy
        return _writable_table_proxy(subtable_path)
    else:
        # Create the table
        with pt.table(str(table_path), table_desc, dminfo=dminfo, ack=False):
            pass

        return _writable_table_proxy(str(table_path))


def _updated_table(table, datasets, columns, descriptor):
    table_proxy = _writable_table_proxy(table)
    table_columns = set(table_proxy.colnames().result())
    missing = set(columns) - table_columns

    # Add missing columns to the table
    if len(missing) > 0:
        # NOTE(sjperkins)
        # Updating a table with new columns with data managers
        # is a little tricky. Trying to update existing data managers
        # seems to incur casacore's internal wrath.
        #
        # Here, we
        # 1. Build a full table description from all variables
        # 2. Take only the column descriptions for the missing variables.
        # 3. Create Data Managers associated with missing variables,
        #    discarding any that currently exist on the table
        builder = descriptor_builder(table, descriptor)
        variables = builder.dataset_variables(datasets)
        default_desc = builder.default_descriptor()
        table_desc = builder.descriptor(variables, default_desc)
        table_desc = {m: table_desc[m] for m in missing}

        # Original Data Manager Groups
        odminfo = {g['NAME'] for g in table_proxy.getdminfo()
                                                 .result()
                                                 .values()}

        # Construct a dminfo object with Data Manager Groups not present
        # on the original dminfo object
        dminfo = {"*%d" % (i + 1): v for i, v
                  in enumerate(builder.dminfo(table_desc).values())
                  if v['NAME'] not in odminfo}

        # Add the columns
        table_proxy.addcols(table_desc, dminfo=dminfo).result()

    return table_proxy


def _add_row_wrapper(table, rows, checkrow=-1):
    startrow = table.nrows()

    if checkrow != -1 and startrow != checkrow:
        raise ValueError("Inconsistent starting row %d %d"
                         % (startrow, checkrow))

    table.addrows(rows)

    return (np.array([[startrow, rows]], dtype=np.int32), None)


def add_row_orders(data, table_proxy, prev=None):
    """
    Adds rows to a table and returns associated row orderings.

    We want to be able to handle undefined row chunk sizes (np.nan) when
    writing dask arrays to a new table. This means that we may not know
    the starting row, or the number of rows in each chunk upfront.
    This poses further challenges when adding new rows to a table.

    This function addresses this by ingesting a chunk of data. From this
    the number of rows in the chunk can be determined and added to the table.
    The starting row and number of rows in the chunk are then returned
    as a result, which is passed as input to the ``add_row_orders`` call,
    operating on an adjacent chunk of row data.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        numpy array from which the number of rows will be derived.
        The first dimension :code:`data.shape[0]`
        should contain the number of rows.
    table_proxy : :class:`daskms.table_proxy.TableProxy`
        Table Proxy object
    prev : tuple or None
        Previous row run array. This argument serves two purposes:

        1. It is used to determine the *starting row* of the current
           row ordering.
        2. When this function is embedded in a dask graph, it establishes a
           dependency on the dask task which creates the previous rows.

        Defaults to ``None``. If ``None``, assumes we're adding rows to
        an empty table. :code:`([[0, rows]], None)` is returned.

        If a :code:`(row_run, resort)` tuple, the first entry in
        the starting row of the previous ``row_run`` is added
        to it's length to produce the starting row of the current row run.

    Returns
    -------
    :class:`numpy.ndarray`
        Row runs of shape :code:`(nruns, 2)` where the first component
        contains the starting row and the last, the number of rows.
    None
        Indicate that row resorting should not occur by default
    """
    rows = data.shape[0]

    # This is the first link in the chain
    if prev is None:
        return (table_proxy.submit(_add_row_wrapper,
                                   WRITELOCK, rows, -1)
                .result())
    else:
        # There's a previous link in the chain
        prev_runs, _ = prev
        startrow = prev_runs.sum()

        return (table_proxy.submit(_add_row_wrapper,
                                   WRITELOCK, rows, startrow)
                .result())


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
            if not dims[0] == 'row':
                continue

            found = True
            token = dask.base.tokenize(array)
            name = '-'.join(('add-rows', str(di), token))
            layers = {}

            for b in range(array.numblocks[0]):
                key = (name, b)
                array_key = (array.name, b) + (0,)*(array.ndim - 1)
                layers[key] = (add_row_orders, array_key,
                               table_proxy, prev_key)
                prev_key = key

            graph = HighLevelGraph.from_collections(name, layers,
                                                    prev_deps + [array])
            chunks = (array.chunks[0],)
            row_adds = da.Array(graph, name, chunks, dtype=np.object)
            row_add_ops.append(row_adds)
            prev_deps = [row_adds]

            break

        if not found:
            raise ValueError("Couldn't find an array with "
                             "which to establish a row ordering "
                             "in dataset %d" % di)

    return row_add_ops


def _write_datasets(table, table_proxy, datasets, columns, descriptor,
                    table_keywords, column_keywords):
    _, table_name, subtable = table_path_split(table)
    table_name = '::'.join((table_name, subtable)) if subtable else table_name
    writes = []
    row_orders = []

    # Put table and column keywords
    table_proxy.submit(_put_keywords, WRITELOCK,
                       table_keywords, column_keywords).result()

    # Sort datasets on (not has "ROWID", index) such that
    # datasets with ROWID's are handled first, while
    # those without (which imply appends to the MS)
    # are handled last
    sorted_datasets = sorted(enumerate(datasets),
                             key=lambda t: ("ROWID" not in t[1].data_vars,
                                            t[0]))

    # Establish row orders for each dataset
    for di, ds in sorted_datasets:
        try:
            rowid = ds.ROWID.data
        except AttributeError:
            # No ROWID's, assume they're missing from the table
            # and remaining datasets. Generate addrows
            # NOTE(sjperkins)
            # This could be somewhat brittle, but exists to
            # update of MS subtables once they've been
            # created (empty) along with the main MS by a call to default_ms.
            # Users could also it to append rows to an existing table.
            # An xds_append_to_table is probably the correct solution...
            last_datasets = datasets[di:]
            last_row_orders = add_row_order_factory(table_proxy, last_datasets)
            row_orders.extend(last_row_orders)
            # We have established row orders for all datasets
            # at this point, quit the loop
            break
        else:
            # Generate row orderings from existing row IDs
            row_order = rowid.map_blocks(row_run_factory,
                                         sort_dir="write",
                                         dtype=np.object)
            row_orders.append(row_order)

    assert len(row_orders) == len(datasets)

    for (di, ds), row_order in zip(sorted_datasets, row_orders):
        data_vars = ds.data_vars

        # Generate a dask array for each column
        for column in columns:
            try:
                variable = data_vars[column]
            except KeyError:
                log.warning("Ignoring '%s' not present "
                            "on dataset %d" % (column, di))
                continue
            else:
                full_dims = variable.dims
                array = variable.data

            args = [row_order, ("row",)]

            # We only need to pass in dimension extent arrays if
            # there is more than one chunk in any of the non-row columns.
            # In that case, we can putcol, otherwise putcolslice is required
            if not all(len(c) == 1 for c in array.chunks[1:]):
                # Add extent arrays
                for d, c in zip(full_dims[1:], array.chunks[1:]):
                    args.append(dim_extents_array(d, c))
                    args.append((d,))

            # Add other variables
            args.extend([table_proxy, None,
                         column, None,
                         array, full_dims])

            # Name of the dask array representing this column
            token = dask.base.tokenize(di, args)
            name = "-".join((table_name, 'write', column, token))

            write_col = da.blockwise(putter_wrapper, full_dims,
                                     *args,
                                     # All dims shrink to 1,
                                     # a single bool is returned
                                     adjust_chunks={d: 1 for d in full_dims},
                                     name=name,
                                     align_arrays=False,
                                     dtype=np.bool)

            writes.append(write_col.ravel())

    if len(writes) == 0:
        return da.full(1, True, dtype=np.bool)

    return da.concatenate(writes)


DELKW = object()


def _put_keywords(table, table_keywords, column_keywords):
    if table_keywords is not None:
        for k, v in table_keywords.items():
            if v == DELKW:
                table.removekeyword(k)
            else:
                table.putkeyword(k, v)

    if column_keywords is not None:
        for column, keywords in column_keywords.items():
            for k, v in keywords.items():
                if v == DELKW:
                    table.removecolkeyword(column, k)
                else:
                    table.putcolkeyword(column, k, v)

    return True


def write_datasets(table, datasets, columns, descriptor=None,
                   table_keywords=None, column_keywords=None):
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
        table_proxy = _create_table(table, datasets, columns, descriptor)
    else:
        table_proxy = _updated_table(table, datasets, columns, descriptor)

    return _write_datasets(table, table_proxy, datasets, columns,
                           descriptor=descriptor,
                           table_keywords=table_keywords,
                           column_keywords=column_keywords)
