API
===

Reading from Tables
-------------------

.. autofunction:: daskms.xds_from_ms

.. autofunction:: daskms.xds_from_table

Writing to Tables
-----------------

.. autofunction:: daskms.xds_to_table

Variables and Datasets
-----------------------

.. autoclass:: daskms.Variable
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

Data Column Expressions
-----------------------

.. autofunction:: daskms.expressions.data_column_expr

Patterns
--------

.. autoclass:: daskms.patterns.Multiton
    :exclude-members: __call__, mro
.. autoclass:: daskms.patterns.LazyProxy
.. autoclass:: daskms.patterns.LazyProxyMultiton
