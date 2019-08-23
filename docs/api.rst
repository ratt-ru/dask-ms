API
===

Reading from Tables
-------------------

.. autofunction:: daskms.xds_from_ms

.. autofunction:: daskms.xds_from_table

Writing to Tables
-----------------

.. autofunction:: daskms.xds_to_table

DataArrays and Datasets
-----------------------

.. autoclass:: daskms.DataArray
    :members:

    .. attribute:: dims

        Dimension schema. :code:`("row", "chan", "corr")` for e.g.

    .. attribute:: data

        Array

    .. attribute:: attrs

        Array metadata dictionary

.. autoclass:: daskms.Dataset
    :members:


    .. automethod:: __init__


TableProxies
------------

.. autoclass:: daskms.TableProxy
    :members:
