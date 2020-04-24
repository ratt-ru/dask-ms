Python Multi-dimensional Array Libraries
----------------------------------------

Python has a rich scientific computing library ecosystem.
The ecosystem's linchpin is `Numpy <https://www.numpy.org/>`_,
which provides accelerated C and FORTRAN operations on multi-dimensional
arrays.

Other libraries such as `Scipy <https://scipy.org/>`_ and
`Astropy <https://www.astropy.org>`_ build on top of Numpy.


CASA Support for NumPy
~~~~~~~~~~~~~~~~~~~~~~

CASA Supports reading and writing Table data into
Numpy arrays via the `python-casacore
<https://github.com/casacore/python-casacore>`_ library.


.. testcode::

    import pyrap.tables as pt
    from daskms.example_data import example_ms

    ms_filename = example_ms()

    with pt.table(ms_filename) as T:
        ddid = T.getcol("DATA_DESC_ID")
        print(ddid)
        print(type(ddid))

produces the following output:

.. testoutput::

    Successful readonly open of default-locked table /tmp/tmp7wkejl07.ms: 22 columns, 10 rows
    [0 0 0 0 1 1 1 1 1 1]
    <class 'numpy.ndarray'>


Specific row ranges can be requested:

.. testcode::

    with pt.table(ms_filename) as T:
        print(T.getcol("DATA_DESC_ID", startrow=2, nrow=4))

.. testoutput::

    [0 0 1 1]

If we wish to arbitrarily access variably shaped data, such
as can be present in the DATA column, `getcol` cannot be (simply)
be used as it is not possible to return a single, fixed shape,
numpy array representing all of this data.

Instead we must make a variably shaped data request via `getvarcol`.:

.. testcode::

    from pprint import pprint

    with pt.table(ms_filename) as T:
        data = T.getvarcol("DATA")
        pprint({k: v.shape for k, v in data.items()})

This produces a dictionary containing variably shaped numpy arrays
for each row, rather than a single array produced by `getcol`:

.. testoutput::

    {'r1': (1, 16, 4),
     'r2': (1, 16, 4),
     'r3': (1, 16, 4),
     'r4': (1, 16, 4),
     'r5': (1, 32, 2),
     'r6': (1, 32, 2),
     'r7': (1, 32, 2),
     'r8': (1, 32, 2),
     'r9': (1, 32, 2),
     'r10': (1, 32, 2)}

However, if we know the first four rows (DATA_DESC_ID = 0) and last
six rows (DATA_DESC_ID = 1) all have the same shape, we can request
data with `getcol`:

.. testcode::

    with pt.table(ms_filename) as T:
        # DATA_DESC_ID = 0 (4 rows, 16 channels, 4 correlations)
        print(T.getcol("DATA", startrow=0, nrow=4).shape)
        # DATA_DESC_ID = 1 (6 rows, 32 channels, 2 correlations)
        print(T.getcol("DATA", startrow=4, nrow=6).shape)


.. testoutput::

    (4, 16, 4)
    (6, 32, 2)


Consult the `python-casacore
<https://github.com/casacore/python-casacore>`_ library for further
information.

Dask
~~~~

`dask <https://dask.pydata.org>`_ is a general
Python parallel programming framework that can distribute work
over multiple cores and nodes. The
`dask Array API <https://docs.dask.org/en/latest/array.html>`_
provides an interface that mimic's that of Numpy, while conceptually
dividing the underlying data into chunks on which operations are
executed in parallel.

The purpose of dask-ms is to expose CASA Table Column data to
the user as dask arrays in order to facilitate parallel programming
of Radio Astronomy Algorithms.

Xarray
~~~~~~

`xarray <https://xarray.pydata.org>`_ groups logically related
numpy and dask arrays into Datasets. Associated dimensions on multiple
arrays can be related to each other, enabling rich data science
applications.

For example, using our example Measurement Set we can do the following:


.. testcode::

    from daskms import xds_from_ms
    from daskms.example_data import example_ms

    datasets = xds_from_ms(example_ms())
    print(datasets)

produces a list of two datasets:

.. testoutput::

    [
        <xarray.Dataset>
         Dimensions:         (chan: 16, corr: 4, row: 4, uvw: 3)
         Coordinates:
             ROWID           (row) int32 dask.array<shape=(4,), chunksize=(4,)>
         Dimensions without coordinates: chan, corr, row, uvw
         Data variables:
             UVW             (row, uvw) float64 dask.array<shape=(4, 3), chunksize=(4, 3)>
             TIME            (row) float64 dask.array<shape=(4,), chunksize=(4,)>
             ANTENNA1        (row) int32 dask.array<shape=(4,), chunksize=(4,)>
             ANTENNA2        (row) int32 dask.array<shape=(4,), chunksize=(4,)>
             DATA            (row, chan, corr) complex64 dask.array<shape=(4, 16, 4), chunksize=(4, 16, 4)>
         Attributes:
             FIELD_ID:      0
             DATA_DESC_ID:  0,

        <xarray.Dataset>
         Dimensions:         (chan: 32, corr: 2, row: 6, uvw: 3)
         Coordinates:
             ROWID           (row) int32 dask.array<shape=(6,), chunksize=(6,)>
         Dimensions without coordinates: chan, corr, row, uvw
         Data variables:
             UVW             (row, uvw) float64 dask.array<shape=(6, 3), chunksize=(6, 3)>
             TIME            (row) float64 dask.array<shape=(6,), chunksize=(6,)>
             ANTENNA1        (row) int32 dask.array<shape=(6,), chunksize=(6,)>
             ANTENNA2        (row) int32 dask.array<shape=(6,), chunksize=(6,)>
             DATA            (row, chan, corr) complex64 dask.array<shape=(6, 32, 2), chunksize=(6, 32, 2)>
         Attributes:
             FIELD_ID:      0
             DATA_DESC_ID:  1
    ]

Keen-eyed readers will note that the first dataset has 4 rows,
16 channels, 4 correlations and DATA_DESC_ID of 0, while the second has
6 rows, 32 channels, 2 correlations and a DATA_DESC_ID of 1.
Here, rows with the same DATA_DESC_ID have been grouped together
into single dataset allowing a column that, while variably shaped,
has fixed shapes for the same DATA_DESC_ID.

The datasets are also grouped on FIELD_ID, but only one FIELD is present
in this dataset.
