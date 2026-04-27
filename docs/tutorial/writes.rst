Writing Datasets
----------------

Creating, Updating or Appending to a Table or Measurement Set is accomplished
through the use of the :func:`~daskms.xds_to_table` and the presence
or absence of the ``ROWID`` coordinate on a
Dataset (See :ref:`row-id-coordinates`).


The pattern for writing a writing a dataset is as follows:

.. doctest::

    >>> from daskms import xds_to_table
    >>> writes = xds_to_table(datasets, "TEST.MS", ["DATA", "BITFLAG"])
    >>> dask.compute(writes)


In the above example, given a list of ``datasets``, the
``DATA`` and ``BITFLAG`` columns are written to the ``TEST.MS`` table.

.. note::

    "ALL" can be supplied to the column argument to specify
    that all arrays should be written to the table. However,
    it is advisable to explicitly specify which columns to write
    to avoid accidentally overwriting data or or performing
    unnecessary writes.


.. _update-append-rows:

Updating/Appending Rows
~~~~~~~~~~~~~~~~~~~~~~~

The presence of ``ROWID`` coordinates on each of the ``datasets`` provided
to :func:`~daskms.xds_to_table` governs whether the function will
update or append rows to a table.

If the ``ROWID`` coordinate is *present* on a dataset, it will be used
to update existing rows in the dataset. By contrast, the *absence* of
``ROWID`` will cause rows to be appended to the table.

The following Dataset without ``ROWID`` creates a new table from scratch.

.. doctest::

    >>> import dask
    >>> import dask.array as da
    >>> from daskms import Dataset
    >>> # Create Dataset Variables
    >>> data_vars = {
        'DATA_DESC_ID': (("row",), da.zeros(10, chunks=2)),
        'DATA': (("row", "chan", "corr"), da.zeros((10, 16, 4), chunks=(2, 16, 4))
    }
    >>> # Write dataset to table
    >>> writes = xds_to_table([Dataset(data_vars)], "test.table", "ALL")
    >>> dask.compute(writes)


It is perfectly possible to combine the two operations by submitting
multiple datasets, some of which contain ``ROWID`` coordinates
while others do not.

.. doctest::

    >>> import dask
    >>> from daskms import xds_from_ms, Dataset
    >>> from daskms.example_data import example_ms
    >>>
    >>> # Create example Measurement Set and read datasets
    >>> ms = example_ms()
    >>> datasets = xds_from_ms(ms)
    >>> # Add last Dataset to table using variables only (no ROWID coordinate)
    >>> new_ds = Dataset(datasets[-1].data_vars)
    >>> datasets.append(new_ds)
    >>>
    >>> # Write datasets back to Measurement Set
    >>> writes = xds_to_table(datasets, ms, "ALL")
    >>> dask.compute(writes)


In these cases it is *strongly* suggested that
the datasets representing updates are generated from
:func:`~daskms.xds_from_table` as this will ensure that the correct
rows are referenced on the dataset. Data from datasets representing
appends will always be added to the end of the table.

Note that it may be also be desirable for appended rows to
have an ordering similar to those of the updated rows, as described
in :ref:`read-sorting`. It is currently the user's responsibility to
achieve this.

Updating/Adding Columns
~~~~~~~~~~~~~~~~~~~~~~~

If a dataset array is present as a column in the dataset, the column will be updated.
By contrast, a missing column will lead cause :func:`~daskms.xds_to_table`
to infer a CASA column descriptor, add the column to the table and then write
the array to it.

.. doctest::

    >>> from daskms import xds_from_ms
    >>> from daskms.example_data import example_ms
    >>>
    >>> ms = example_ms()
    >>> datasets = xds_from_ms(ms)
    >>>
    >>> # Add BITFLAG data to datasets
    >>> for i, ds in enumerate(datasets):
    >>>     datasets[i] = ds.assign(BITFLAG=(("row", "chan", "corr",
                                              da.zeros_like(ds.DATA.data))))
    >>>
    >>> # Write data back to ms
    >>> writes = xds_to_table(datasets, ms, ["BITFLAG"])
    >>> dask.compute(writes)


Creating and updating the Measurement Set and it's sub-tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case of the Measurement Set and it's sub-tables,
care is taken to ensure that

1. Required columns are added.
2. Required columns conform to the `Measurement Set v2.0 Specification
   <https://casacore.github.io/casacore-notes/229.html>`_.

This means that, for example, if you have a UVW array
with a non-standard shape ([4]) and type (float), the UVW column
will still be created the shape ([3]) and type (double)
mandated by the MSv2.0 spec.

The above also applies to the following optional columns in the MSv2.0:

+-----------------+
| DATA            |
+-----------------+
| MODEL_DATA      |
+-----------------+
| CORRECTED_DATA  |
+-----------------+
| WEIGHT_SPECTRUM |
+-----------------+
| SIGMA_SPECTRUM  |
+-----------------+
| IMAGING_WEIGHTS |
+-----------------+

Other optional MSv2.0 columns can easily be supported.

This behaviour is triggered whenever the ``table_name`` ends
with lower or uppercase ``.MS`` in the case of the main
Measurement Set table:

.. doctest::

    >>> xds_to_table(datasets, "test.ms", ["DATA", "BITFLAG"])

or when it ends with with ``::subtablename`` in the case of a subtable:

.. doctest::

    >>> xds_to_table(datasets, "test.ms::SPECTRAL_WINDOW", ["CHAN_FREQ"])

Respect the standard naming conventions and you'll be fine.


Creating Sub-tables
~~~~~~~~~~~~~~~~~~~

It is possible for sub-tables to be added to a table.
For example, the SOURCE table is an optional table that may or may not
be present on the Measurement Set

The following convention specifies that the ``SOURCE`` sub-table
of ``TEST.MS`` should be created:

.. doctest::

    >>> writes = xds_to_table(source_dataset,
                              "~/data/TEST.MS::SOURCE",
                              columns="ALL")

``xds_to_table`` will also created the ``"Table: ~/data/TEST.MS/SOURCE"``
keyword in ``TEST.MS`` linking it with the ``SOURCE`` sub-table.

.. warning::

    As discussed in :ref:`read-opening-sub-tables`, it is advisable to use the
    `::` scope operator so that dask-ms understands the link between the
    main table and the sub-table. The following will create a SOURCE table
    but will not create a link between the table and the sub-table:

    .. doctest::

        >>> writes = xds_to_table(source_dataset,
                                  "~/data/TEST.MS/SOURCE",
                                  columns="ALL")

Keywords
~~~~~~~~

Keywords can be added to the target table and columns:

.. doctest::

    >>> xds_to_table(datasets, "test.ms", [],
                     table_keywords={"foo":"bar"},
                     column_keywords={"DATA": {"foo": "bar"}})


.. _array-api-writes:

Writing Arrays from Array API Compatible Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dask-ms supports writing arrays from libraries that implement the
`Python Array API standard (2021.12+) <https://data-apis.org/array-api/latest/>`_,
including JAX, CuPy, and PyTorch.  Arrays are converted to NumPy inside
each write task, so no changes to the :func:`~daskms.xds_to_table` call
are needed.  GPU arrays are transferred to CPU automatically, one chunk
at a time, avoiding a full device-to-host copy at graph-construction time.

The only practical difference between libraries is **how you wrap the
source array in a dask array**:

- **JAX and CuPy** expose NumPy-compatible dtype objects, so
  :func:`dask.array.from_array` works without any extra arguments.
- **PyTorch** uses its own dtype objects (``torch.complex64``, etc.)
  which are not recognised by NumPy at graph-construction time, so
  :func:`dask.array.from_delayed` with an explicit ``dtype`` is required.

JAX
^^^

:func:`dask.array.from_array` works directly with ``jax.Array`` objects.
Each chunk will be a ``jax.Array``; dask-ms converts it to NumPy inside
the write task.

.. code-block:: python

    import dask
    import dask.array as da
    import jax.numpy as jnp
    from daskms import xds_from_ms, xds_to_table

    ms = "path/to/data.ms"
    ds = xds_from_ms(ms, columns=["DATA"], group_cols=[],
                     chunks={"row": 100})[0]
    dims, chunks = ds.sizes, ds.chunks

    jax_data = jnp.zeros(
        (dims["row"], dims["chan"], dims["corr"]), dtype=jnp.complex64
    )
    da_data = da.from_array(
        jax_data, chunks=(chunks["row"], dims["chan"], dims["corr"])
    )
    new_ds = ds.assign(DATA=(("row", "chan", "corr"), da_data))
    dask.compute(xds_to_table(new_ds, ms, ["DATA"]))

This pattern also applies when ``jax.Array`` objects are produced
per-chunk, for example from a :func:`dask.array.map_blocks` call over a
JAX-jitted function:

.. code-block:: python

    import jax

    @jax.jit
    def predict(uvw_chunk):
        ...  # returns a jax.Array of shape (row, chan, corr)

    da_data = da.map_blocks(
        predict, ds.UVW.data,
        dtype="complex64",
        new_axis=[1, 2],
        chunks=(chunks["row"], dims["chan"], dims["corr"]),
    )

CuPy
^^^^

CuPy arrays live on the GPU.  :func:`dask.array.from_array` introspects
the dtype correctly, and dask-ms transfers each chunk to CPU inside the
write task via ``cupy.ndarray.get()``.

.. code-block:: python

    import dask
    import dask.array as da
    import cupy as cp
    from daskms import xds_from_ms, xds_to_table

    ms = "path/to/data.ms"
    ds = xds_from_ms(ms, columns=["DATA"], group_cols=[],
                     chunks={"row": 100})[0]
    dims, chunks = ds.sizes, ds.chunks

    cupy_data = cp.zeros(
        (dims["row"], dims["chan"], dims["corr"]), dtype=cp.complex64
    )
    da_data = da.from_array(
        cupy_data, chunks=(chunks["row"], dims["chan"], dims["corr"])
    )
    new_ds = ds.assign(DATA=(("row", "chan", "corr"), da_data))
    dask.compute(xds_to_table(new_ds, ms, ["DATA"]))

PyTorch
^^^^^^^

PyTorch tensor dtype objects (``torch.complex64``, etc.) are not
NumPy-compatible, so :func:`dask.array.from_array` cannot introspect the
dtype at graph-construction time.  Use :func:`dask.array.from_delayed`
with an explicit NumPy ``dtype`` string instead, building one delayed
chunk per row-chunk:

.. code-block:: python

    import dask
    import dask.array as da
    import torch
    from daskms import xds_from_ms, xds_to_table

    ms = "path/to/data.ms"
    ds = xds_from_ms(ms, columns=["DATA"], group_cols=[],
                     chunks={"row": 100})[0]
    dims, chunks = ds.sizes, ds.chunks

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_data = torch.zeros(
        dims["row"], dims["chan"], dims["corr"],
        dtype=torch.complex64, device=device,
    )

    parts, row = [], 0
    for rc in chunks["row"]:
        chunk = torch_data[row: row + rc]
        parts.append(
            da.from_delayed(
                dask.delayed(chunk),
                shape=(rc, dims["chan"], dims["corr"]),
                dtype="complex64",  # NumPy dtype — no torch dependency here
            )
        )
        row += rc
    da_data = da.concatenate(parts, axis=0)

    new_ds = ds.assign(DATA=(("row", "chan", "corr"), da_data))
    dask.compute(xds_to_table(new_ds, ms, ["DATA"]))

For CUDA tensors, the device-to-host transfer (``tensor.to("cpu")``) is
performed inside each write task.  For CPU tensors no copy is made at the
dask layer.
