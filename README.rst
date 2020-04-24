================================
xarray Datasets from CASA Tables
================================

.. image:: https://img.shields.io/pypi/v/dask-ms.svg
        :target: https://pypi.python.org/pypi/dask-ms

.. image:: https://img.shields.io/travis/ska-sa/dask-ms.svg
        :target: https://travis-ci.org/ska-sa/dask-ms

.. image:: https://readthedocs.org/projects/dask-ms/badge/?version=latest
        :target: https://dask-ms.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Constructs xarray_ ``Datasets`` from CASA Tables via python-casacore_.
The ``Variables`` contained in the ``Dataset`` are dask_ arrays backed by
deferred calls to :code:`pyrap.tables.table.getcol`.

Supports writing ``Variables`` back to the respective column in the Table.

The intention behind this package is to support the Measurement Set as
a data source and sink for the purposes of writing parallel, distributed
Radio Astronomy algorithms.

Installation
============

To install with xarray_ support:

.. code-block:: bash

  $ pip install dask-ms[xarray]

Without xarray_ similar, but reduced Dataset functionality is replicated
in dask-ms itself. Expert users may wish to use this option to reduce
python package dependencies.

.. code-block:: bash

  $ pip install dask-ms


Documentation
=============

https://dask-ms.readthedocs.io

Gitter Page
===========

https://gitter.im/dask-ms/community

Example Usage
=============


.. code-block:: python

    import dask.array as da
    from daskms import xds_from_table, xds_to_table

    # Create xarray datasets from Measurement Set "WSRT.MS"
    ds = xds_from_table("WSRT.MS")
    # Set the flag Variable on first Dataset to it's inverse
    ds[0]['flag'] = (ds[0].flag.dims, da.logical_not(ds[0].flag))
    # Write the flag column back to the Measurement Set
    xds_to_table(ds, "WSRT.MS", "FLAG").compute()

    print ds

  [<xarray.Dataset>
   Dimensions:         (chan: 64, corr: 4, row: 6552, uvw: 3)
   Coordinates:
       ROWID           (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
   Dimensions without coordinates: chan, corr, row, uvw
   Data variables:
       IMAGING_WEIGHT  (row, chan) float32 dask.array<shape=(6552, 64), chunksize=(6552, 64)>
       ANTENNA1        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       STATE_ID        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       EXPOSURE        (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
       MODEL_DATA      (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
       FLAG_ROW        (row) bool dask.array<shape=(6552,), chunksize=(6552,)>
       CORRECTED_DATA  (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
       PROCESSOR_ID    (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       WEIGHT          (row, corr) float32 dask.array<shape=(6552, 4), chunksize=(6552, 4)>
       FLAG            (row, chan, corr) bool dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
       TIME            (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
       SIGMA           (row, corr) float32 dask.array<shape=(6552, 4), chunksize=(6552, 4)>
       SCAN_NUMBER     (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       INTERVAL        (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
       OBSERVATION_ID  (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       TIME_CENTROID   (row) float64 dask.array<shape=(6552,), chunksize=(6552,)>
       ARRAY_ID        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       ANTENNA2        (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       DATA            (row, chan, corr) complex64 dask.array<shape=(6552, 64, 4), chunksize=(6552, 64, 4)>
       FEED1           (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       FEED2           (row) int32 dask.array<shape=(6552,), chunksize=(6552,)>
       UVW             (row, uvw) float64 dask.array<shape=(6552, 3), chunksize=(6552, 3)>
   Attributes:
       FIELD_ID:      0
       DATA_DESC_ID:  0]

-----------
Limitations
-----------

1. Many Measurement Sets columns are defined as variably shaped,
   but the actual data is fixed.
   dask-ms_ will infer the shape of the
   data from the first row and must be consistent
   with that of other rows.
   For example, this may be issue where multiple Spectral Windows
   are present in the Measurement Set with differing channels
   per SPW.

   dask-ms_ works around this by partitioning the
   Measurement Set into multiple datasets.
   The first row's shape is used to infer the shape of the partition.
   Thus, in the case of multiple Spectral Window's, we can partition
   the Measurement Set by DATA_DESC_ID to create a dataset for
   each Spectral Window.

.. _dask: https://dask.pydata.org
.. _dask-ms: https://github.com/ska-sa/dask-ms
.. _xarray: https://xarray.pydata.org
.. _python-casacore: https://github.com/casacore/python-casacore
