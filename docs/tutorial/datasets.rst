Basic Dataset Manipulation
--------------------------

This section describes Dataset operations available on
:class:`~daskms.Dataset` which exists to re-implement
basic operations present on xarray Datasets.
These should be sufficient for application developers.

Those wanting extended xarray Dataset functionality can simply install
`xarray <https://xarray.pydata.org>`_ as the standard dask-ms
``xds_from*`` and ``xds_to*`` functions return and accept xarray Datasets.

Datasets logically group arrays together into a single structure.

Dataset Variables
~~~~~~~~~~~~~~~~~

:class:`~daskms.Variable`'s' are represented by a tuple
of two or three variables.
They have the form :code:`(dims, array[, attrs])`.

- ``dim`` is a dimension schema. The first entry in ``dim`` must always
  be ``"row"``.
- ``array`` should be a `dask` or `numpy` array.
- ``attrs`` is optional and should be a dictionary containing metadata.

.. code-block:: python

    IMAGING_WEIGHT = (("row", "chan"), np.zeros(10, 16), {"keywords": test})

Creating Datasets
~~~~~~~~~~~~~~~~~

Set up imports and define some dimension chunks and sizes:

.. code-block:: python

    import numpy as np
    from daskms import Dataset
    # Define a chunking schema
    chunks = {'row': (2, 2, 2, 2, 2), 'chan': (16, 16), "corr": (4,)}
    # Figure out dimension sizes
    row = sum(chunks['row'])
    chan = sum(chunks['chan'])
    corr = sum(chunks['corr'])

Next, create some dask arrays that we will place on our Dataset

.. code-block:: python

    # Define a data descriptor array
    ddid = da.ones(row, chunks=chunks['row'])

    # Define some visibilities
    vis_chunks = (chunks['row'], chunks['chan'], chunks['corr'])
    data = (da.random.random((row, chan, corr), chunks=vis_chunks) +
            da.random.random((row, chan, corr), chunks=vis_chunks)*1j)


Next, create the dataset by assigning variable dictionaries.
They have the form :code:`{name: (dims, array[, attrs])}`

The :class:`~daskms.Dataset` can also be assigned coordinates and attributes
via the ``coords`` and ``attrs`` argument to the constructor.

.. note::

    The ROWID coordinate is not normally assigned when creating
    a Dataset from scratch and is shown here for illustrating
    how to set coordinates.
    See :ref:`update-append-rows` for further information on
    standard use of the ROWID array.

.. code-block:: python

    # Data Variable dictionary
    data_vars = {
        'DATA_DESC_ID' : (("row"), ddid, {'keywords': 'test'})
        'DATA': (("row", "chan", "corr"), data)}

    # Coordinate dictionary
    coords = {'ROWID': (("row"), rowid)}

    # Create the dataset
    ds = Dataset(data_vars, attrs={'observer': 'hugo'}, coords=coords})



Modifying Datasets
~~~~~~~~~~~~~~~~~~

We can assign new variables to our Dataset

.. code-block:: python

    bitflag = da.ones((row, chan, corr), chunks=vis_chunks)

    new_ds = ds.assign(BITFLAG=(("row", "chan", "corr"), bitflag))
