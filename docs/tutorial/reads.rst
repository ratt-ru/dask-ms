Reading Datasets
----------------

Basic Use
~~~~~~~~~

There are two methods for creating Datasets from a CASA Table or
Measurement Set. :func:`~daskms.xds_from_table` handles general tables while
:func:`~daskms.xds_from_ms` handles Measurement Sets specifically.
We will use the two interchangeably.

Calling either of these two functions will produce a list of datasets:

.. doctest::

    >>> from daskms import xds_from_ms
    >>> datasets = xds_from_ms("~/data/TEST.MS")

    >>> # Print list of datasets
    >>> print(datasets)

    [<xarray.Dataset>
     Dimensions:         (chan: 64, corr: 4, row: 6552, uvw: 3)
     Coordinates:
         ROWID           (row) object dask.array<shape=(6552,), chunksize=(6552,)>
     Dimensions without coordinates: chan, corr, row, uvw
     Data variables:
         ANTENNA1        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         ANTENNA2        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         ARRAY_ID        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         CORRECTED_DATA  (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
         DATA            (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
         DATA_DESC_ID    (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         EXPOSURE        (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
         FEED1           (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         FEED2           (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         FIELD_ID        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         FLAG            (row, chan, corr) bool dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
         FLAG_ROW        (row) bool dask.array<shape=(6552,), chunksize=(6552,)>
         IMAGING_WEIGHT  (row, chan) float32 dask.array<shape=(6552, 64), chunksize=(6552, 64)>
         INTERVAL        (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
         MODEL_DATA      (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
         OBSERVATION_ID  (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         PROCESSOR_ID    (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         SCAN_NUMBER     (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         SIGMA           (row, corr) float32 dask.array<shape=(6552, 4), chunksize=(6552, 4)>
         STATE_ID        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
         TIME            (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
         TIME_CENTROID   (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
         UVW             (row, uvw) float64 dask.array<shape=(6552, 3), chunksize=(6552, 3)>
         WEIGHT          (row, corr) float32 dask.array<shape=(6552, 4), chunksize=(6552, 4)>]


The arrays stored in the dataset are :class:`xarray.DataArray` objects, which
wrap numpy or dask arrays and store related metadata. For example:

.. doctest::

    >>> # Print the xarray.DataArray
    >>> print(datasets[0].TIME)

    <xarray.DataArray 'TIME' (row: 6552)>
    dask.array<shape=(6552,), dtype=float64, chunksize=(6552,)>
    Coordinates:
        ROWID    (row) object dask.array<shape=(6552,), chunksize=(6552,)>
    Dimensions without coordinates: row
    Attributes:
        keywords:  {'QuantumUnits': ['s'], 'MEASINFO': {'type': 'epoch', 'Ref': '...

Access array dimension schema:

.. doctest::

    >>> # Print the xarray.DataArray.dims
    >>> print(datasets[0].TIME.dims)

    ('row',)

Access internal numpy/dask array:

.. doctest::

    >>> # Print the dask array wrapped by xarray.DataArray
    >>> print(datasets[0].TIME.data)

    dask.array<TEST.MS-TIME, shape=(6552,), dtype=float64, chunksize=(6552,)>

Access array attribute dictionary:

.. doctest::

    >>> # Print attributes associated with the xarray.DataArray
    >>> print(dict(datasets[0].TIME.attrs))

    {'keywords': {'QuantumUnits': ['s'],
                  'MEASINFO': {'type': 'epoch', 'Ref': 'UTC'}}}


.. _read-opening-sub-tables:

Opening Sub-tables
~~~~~~~~~~~~~~~~~~

CASA Tables can also have sub-tables associated with them.
For example, the Measurement Set has ANTENNA, SPECTRAL_WINDOW
and DATA_DESCRIPTION sub-tables.

``::``, the traditional scope operator used by
`Taql <https://casacore.github.io/casacore-notes/199.html>`_
used to reference the sub-tables of a table, is
understood by python-casacore and dask-ms.
The following convention specifies that the ``ANTENNA`` sub-table
of ``TEST.MS`` should be opened:

.. doctest::

    >>> ant_datasets = xds_from_table("~/data/TEST.MS::ANTENNA")

It is recommended that the ``TEST.MS::ANTENNA`` convention be
followed as it makes the link between the table and sub-table clear to dask-ms.

Alternatively, as sub-tables are simply stored as sub-directories
of the main table, it is also possible to reference them as follows:

.. doctest::

    >>> ant_datasets = xds_from_table("~/data/TEST.MS/ANTENNA")


Grouping
~~~~~~~~

As discussed previously we frequently wish to group associated table rows
together. This can be useful in the following cases:

- Group a variably shaped column into rows that share each other's shapes.
  For example, grouping by **DATA_DESC_ID** on a Measurement Set will
  produce datasets whose rows contain the same channels and correlations.
- Logically separate unique column values into separate datasets.
  For example, we may wish to create datasets on unique
  **FIELD_ID** and **SCAN_NUMBER**.


`xds_from_table` takes a `group_cols` argument that specify which columns
will contribute to a grouping. For example:

.. doctest::

    >>> from daskms import xds_from_ms
    >>> group_cols = ["FIELD_ID", "SCAN_NUMBER", "DATA_DESC_ID"]
    >>> datasets = xds_from_ms("~/data/TEST.MS", group_cols=group_cols)

    >>> # Print list of datasets
    >>> print(datasets)
    [<xarray.Dataset>
     Data variables:
         ANTENNA1        (row) int32 dask.array<shape=(128,), chunksize=(128,)>
         ...
    Attributes:
        FIELD_ID:       0
        SCAN_NUMBER:    0
        DATA_DESC_ID:   0,

    <xarray.Dataset>
     Data variables:
         ANTENNA1        (row) int32 dask.array<shape=(164,), chunksize=(164,)>
         ...
    Attributes:
        FIELD_ID:       0
        SCAN_NUMBER:    0
        DATA_DESC_ID:   1,

    <xarray.Dataset>
     Data variables:
         ANTENNA1        (row) int32 dask.array<shape=(96,), chunksize=(96,)>
         ...
    Attributes:
        FIELD_ID:       0
        SCAN_NUMBER:    1
        DATA_DESC_ID:   1
    ...]


Here, all rows with (FIELD_ID SCAN_NUMBER, DATA_DESC_ID) = (0, 0, 0) are grouped
into the first dataset, (0, 0, 1) into the second, (0, 1, 1) into the third
and so forth. More specifically, a list of datasets containing the
Cartesian product of all unique grouping column values is returned.
Conversely, if `group_cols` is not specified then only a single dataset
is returned.

Grouping by row
+++++++++++++++

Frequently, Measurement Sub-tables will have variably shaped columns,
for example the **SPECTRAL_WINDOW** table, where each row
describes a variable range of frequencies, or the **POLARIZATION** table,
where each row describes a correlation configuration.

In the presence of such variability, it is often useful to group each
row into a separate dataset using the ``__row__`` marker.

.. doctest::

    >>> from daskms import xds_from_table
    >>> datasets = xds_from_ms("~/data/TEST.MS::SPECTRAL_WINDOW", group_cols="__row__")
    >>> print(datasets)
    [<xarray.Dataset>
     Dimensions:          (chan: 64, row: 1)
     Coordinates:
         ROWID            (row) object dask.array<shape=(1,), chunksize=(1,)>
     Dimensions without coordinates: chan, row
     Data variables:
         CHAN_FREQ        (row, chan) float64 dask.array<shape=(1, 64), chunksize=(1, 64)>,
    <xarray.Dataset>
     Dimensions:          (chan: 4096, row: 1)
     Coordinates:
         ROWID            (row) object dask.array<shape=(1,), chunksize=(1,)>
     Dimensions without coordinates: chan, row
     Data variables:
         CHAN_FREQ        (row, chan) float64 dask.array<shape=(1, 4096), chunksize=(1, 4096)>,
    ]

It's often useful to squeeze out the row to just get the channel dimension
in this case:

    >>> datasets[0].CHAN_FREQ.data.squeeze(0).compute()
    array([1.4000625e+09, 1.4001875e+09, 1.4003125e+09, 1.4004375e+09,
       ...
       1.4075625e+09, 1.4076875e+09, 1.4078125e+09, 1.4079375e+09])

Table Joins
+++++++++++

Grouping by row is frequently useful for joining data on sub-tables
with data on the main table. In the following example we group
by **DATA_DESC_ID** and wish to discover the frequency range
and correlation types associated with our visibility data.

First we group the Measurement Set on **DATA_DESC_ID**

.. doctest::

    >>> from daskms import xds_from_table
    >>> # Get Measurement Set datasets, grouped on DATA_DESC_ID
    >>> ms = xds_from_ms("~/data/TEST.MS", group_cols=["DATA_DESC_ID"])


We then create a single dataset from the **DATA_DESCRIPTION** table
and compute its contents (simple indices) upfront.

.. doctest::

    >>> # Get DATA_DESCRIPTION datasets
    >>> ddids = xds_from_table("~/data/TEST.MS::DATA_DESCRIPTION")
    >>> # Convert from dask to numpy arrays
    >>> ddids = ddids.compute()

We now create datasets for each row of the **SPECTRAL_WINDOW** and
**POLARIZATION** tables:

.. doctest::

    >>> # Get SPECTRAL_WINDOW datasets, one per row
    >>> spws = xds_from_table("~/data/TEST.MS::SPECTRAL_WINDOW", group_cols="__row__")
    >>> # Get POLARIZATION datasets, one per row
    >>> pols = xds_from_table("~/data/TEST.MS::POLARIZATION", group_cols="__row__")
    >>>
    >>> for msds in ms:
    >>>     # Get DATA_DESC_ID value for group
    >>>     ddid = msds.attrs['DATA_DESC_ID']
    >>>     # Get SPW index, removing single row dimension
    >>>     spw_id = ddids[ddid].SPECTRAL_WINDOW_ID.data[0]
    >>>     # Get POL index, removing single row dimension
    >>>     pol_id = ddids[ddid].POLARIZATION_ID.data[0]
    >>>     # Get channel frequencies, removing single row dimension
    >>>     chan_freq = spws[spw_id].CHAN_FREQ.data[0]
    >>>     # Get correlation type, removing single row dimension
    >>>     corr_type = pols[pol_id].CORR_TYPE.data[0]


.. _read-sorting:

Sorting
~~~~~~~

Frequently we wish our rows to be ordered according to some sorting
criteria. `index_cols` can be supplied in order to produce this ordering on
the Dataset arrays:

    >>> from daskms import xds_from_table
    >>> # Get Measurement Set datasets, grouped on DATA_DESC_ID and
    >>> # sorted on TIME, ANTENNA1 and ANTENNA2
    >>> ms = xds_from_ms("~/data/TEST.MS", group_cols=["DATA_DESC_ID"].
    >>>                  index_cols=["SCAN_NUMBER", "TIME", "ANTENNA1", "ANTENNA2"])

.. note::

    Care should be taken to ensure that the requested ordering is
    not egregiously different from the way the data is structured internally
    within the table. This structure is usually defined by the order
    data is written to the table. A
    ``["SCAN_NUMBER", TIME", "ANTENNA1", "ANTENNA2"]`` is a fairly natural ordering
    while the reverse is not.

    Unnatural orderings result in non-contiguous row access patterns which
    can badly affect I/O performance. dask-ms attempts to ameliorate this
    by resorting row id's within a dask array chunk to produce
    access patterns that are as contiguous as possible, but this is not
    a panacea.

    The rule of thumb is that the more your ``index_cols`` tends towards a
    lexicographical ordering, the more optimal your table access patterns will be.


 .. _row-id-coordinates:

ROWID Coordinates
~~~~~~~~~~~~~~~~~

Each read dataset has a ``ROWID`` coordinate associated with it.
This is a dask array that associates a ROWID with each ``row``
in the Dataset.

.. doctest::

    >>> from daskms import xds_from_ms
    >>> datasets = xds_from_ms("~/data/TEST.MS")
    >>> print(datasets)

    [<xarray.Dataset>
     Dimensions:         (chan: 64, corr: 4, row: 6552, uvw: 3)
     Coordinates:
         ROWID           (row) object dask.array<shape=(6552,), chunksize=(6552,)>
     Data variables:
        ...
         DATA            (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>

This array is related to the Grouping_ and Sorting_ requested
on the table and will generally be contiguous if the
requested grouping and sorting represents a natural lexicographical ordering.

For example a natural ordering:

.. doctest::

    >>> datasets = xds_from_ms("~/data/TEST.MS")
    >>> print(datasets[0].ROWID.data.compute())
    array([   0,    1,    2, ..., 6549, 6550, 6551])

vs a non-contiguous ordering:

.. doctest::


    >>> datasets = xds_from_ms("~/data/TEST.MS", index_cols=["ANTENNA2", "ANTENNA1", "TIME"])
    >>> print(datasets[0].ROWID.data.compute())
    array([   0,   91,  182, ..., 6369, 6460, 6551])

Internally, it is used to request or supply **ranges** of data from the Table
when reading and writing, respectively.

.. _read-keywords:

Keywords
~~~~~~~~

It is possible to request both the table and column keywords:

.. doctest::

    >>> from daskms import xds_from_ms
    >>> datasets, tabkw, colkw = xds_from_ms("~/data/TEST.MS",
                                             table_keywords=True,
                                             column_keywords=True)
    >>> print(tabkw)
    {'MS_VERSION': 2.0,
     'ANTENNA': 'Table: ~/data/TEST.MS/ANTENNA',
     'DATA_DESCRIPTION': 'Table: ~/data/TEST.MS/DATA_DESCRIPTION',
     ...
     'SPECTRAL_WINDOW': 'Table: ~/data/TEST.MS/SPECTRAL_WINDOW',
     'STATE': 'Table: ~/data/TEST.MS/STATE'}

    >>> print(colkw)
    'UVW': {'QuantumUnits': ['m', 'm', 'm'],
      'MEASINFO': {'type': 'uvw', 'Ref': 'J2000'}},
      ...
     'TIME': {'QuantumUnits': ['s'], 'MEASINFO': {'type': 'epoch', 'Ref': 'UTC'}},
     'TIME_CENTROID': {'QuantumUnits': ['s'],
      'MEASINFO': {'type': 'epoch', 'Ref': 'UTC'}},
     'DATA': {'UNIT': 'Jy'},
     'MODEL_DATA': {'CHANNEL_SELECTION': array([[ 0, 64]], dtype=int32)}}


