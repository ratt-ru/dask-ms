API
===

.. autofunction:: daskms.xds_from_ms

.. autofunction:: daskms.xds_from_table

.. autofunction:: daskms.xds_to_table

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



.. autoclass:: daskms.TableProxy
    :members:
