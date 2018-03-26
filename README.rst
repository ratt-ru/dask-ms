================================
xarray Datasets from CASA Tables
================================

.. image:: https://img.shields.io/pypi/v/xarray-ms.svg
        :target: https://pypi.python.org/pypi/xarray-ms

.. image:: https://img.shields.io/travis/ska-sa/xarray-ms.svg
        :target: https://travis-ci.org/ska-sa/xarray-ms

.. image:: https://readthedocs.org/projects/xarray-ms/badge/?version=latest
        :target: https://xarray-ms.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Constructs xarray_ ``Datasets`` from CASA Tables via python-casacore_.
The ``DataArrays`` contained in the ``Dataset`` are dask_ arrays backed by
deferred calls to :code:`pyrap.tables.table.getcol`.

Supports writing ``DataArrays`` back to the respective column in the Table.

The intention behind this package is to support the Measurement Set as
a data source and sink for the purposes of writing parallel, distributed
Radio Astronomy algorithms.

.. code-block:: python

    import dask.array as da
    from xarrayms import xds_from_table, xds_to_table

    # Create xarray dataset from Measurement Set "WSRT.MS"
    ds = xds_from_table("WSRT.MS")
    # Set the flag DataArray to it's inverse
    ds['flag'] = (ds.flag.dims, da.logical_not(ds.flag))
    # Write the flag column back to the Measurement Set
    xds_to_table(ds, "WSRT.MS", "FLAG").compute()

    print ds

    <xarray.Dataset>
    Dimensions:         ((u,v,w): 3, chan: 64, corr: 4, row: 6552, table_row: 6552)
    Coordinates:
      * row             (row) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
      * table_row       (table_row) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ...
    Dimensions without coordinates: (u,v,w), chan, corr
    Data variables:
        ANTENNA1        (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        ANTENNA2        (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        ARRAY_ID        (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        CORRECTED_DATA  (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(1000, 64, 4)>
        DATA            (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(1000, 64, 4)>
        EXPOSURE        (row) float64 dask.array<shape=(6552,), chunksize=(1000,)>
        FEED1           (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        FEED2           (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        FLAG            (row, chan, corr) bool dask.array<shape=(6552, 64, 4), chunksize=(1000, 64, 4)>
        FLAG_ROW        (row) bool dask.array<shape=(6552,), chunksize=(1000,)>
        IMAGING_WEIGHT  (row, chan) float32 dask.array<shape=(6552, 64), chunksize=(1000, 64)>
        INTERVAL        (row) float64 dask.array<shape=(6552,), chunksize=(1000,)>
        MODEL_DATA      (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(1000, 64, 4)>
        OBSERVATION_ID  (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        PROCESSOR_ID    (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        SCAN_NUMBER     (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        SIGMA           (row, corr) float32 dask.array<shape=(6552, 4), chunksize=(1000, 4)>
        STATE_ID        (row) int32 dask.array<shape=(6552,), chunksize=(1000,)>
        TIME            (row) float64 dask.array<shape=(6552,), chunksize=(1000,)>
        TIME_CENTROID   (row) float64 dask.array<shape=(6552,), chunksize=(1000,)>
        UVW             (row, (u,v,w)) float64 dask.array<shape=(6552, 3), chunksize=(1000, 3)>
        WEIGHT          (row, corr) float32 dask.array<shape=(6552, 4), chunksize=(1000, 4)>
    Attributes:
        FIELD_ID:      0
        DATA_DESC_ID:  0

-------------
Documentation
-------------

https://xarray-ms.readthedocs.io.

-----------
Limitations
-----------

1. Many Measurement Sets columns are defined as variably shaped,
   but the actual data is fixed.
   xarray-ms_ will infer the shape of the
   data from the first row and must be consistent
   with that of other rows.
   For example, this may be issue where multiple Spectral Windows
   are present in the Measurement Set with differing channels
   per SPW.

   xarray-ms_ works around this by partitioning the
   Measurement Set into multiple datasets.
   The first row's shape is used to infer the shape of the partition.
   Thus, in the case of multiple Spectral Window's, we can partition
   the Measurement Set by DATA_DESC_ID to create a dataset for
   each Spectral Window.

.. _dask: https://dask.pydata.org
.. _xarray-ms: https://github.com/ska-sa/xarray-ms
.. _xarray: https://xarray.pydata.org
.. _python-casacore: https://github.com/casacore/python-casacore
