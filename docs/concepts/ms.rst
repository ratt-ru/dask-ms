CASA Tables
-----------

The CASA Table can be viewed as a `Relational Database
<https://en.wikipedia.org/wiki/Relational_database>`_ for storing
Radio Astronomy Data.
Familiarity with `Relational Database Management System
<https://www.tutorialspoint.com/sql/sql-rdbms-concepts.htm>`_ (RDBMS) concepts
is useful when working with CASA tables.
For example, Relational Databases are commonly accessed via the
`Structured Query Language <https://en.wikipedia.org/wiki/SQL>`_  and
the CASA Table system has its own `TAble Query Language (TAQL)
<https://casacore.github.io/casacore-notes/199.html>`_ dialect.


Rows and Columns
~~~~~~~~~~~~~~~~

Briefly, data is stored in a table consisting of rows and columns.
For example, an observation can be represented by the following table:

.. Generated with http://www.tablesgenerator.com/text_tables

+--------------------------------------------------------+
|                        MAIN                            |
+-----+----------+------+----------+----------+----------+
| row | FIELD_ID | TIME | ANTENNA1 | ANTENNA2 | DATA     |
+-----+----------+------+----------+----------+----------+
| 0   | 0        | 0.1  | 0        | 1        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 1   | 0        | 0.1  | 0        | 2        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 2   | 0        | 0.1  | 0        | 3        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 3   | 0        | 0.2  | 0        | 1        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 4   | 0        | 0.2  | 0        | 2        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 5   | 0        | 0.2  | 0        | 3        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 6   | 1        | 0.5  | 0        | 1        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 7   | 1        | 0.5  | 0        | 1        | (4096,4) |
+-----+----------+------+----------+----------+----------+
| 8   | 1        | 0.5  | 0        | 1        | (4096,4) |
+-----+----------+------+----------+----------+----------+

Each row stores a measurement produced by an interferometer
over 4096 channels and 4 correlations in the DATA column.
Other columns identify attributes associated with each measurement.

In the above example, each measurement is uniquely identified by the:

- FIELD_ID: The ID of the observed FIELD
- TIME: The time at which the measurement was taken.
- ANTENNA1, ANTENNA2: The baseline along which the measurement was taken.

The (FIELD_ID, TIME, ANTENNA1, ANTENNA2) values therefore
establish *unique key* for the table, in the Relational Model.

Column Types and Shapes
~~~~~~~~~~~~~~~~~~~~~~~

Columns contain data of a specific type (integer, float, string, complex).

The shape of column data can be configured to be
highly constrained (all rows have a fixed shape)
or highly variable (many rows have many shapes).

For example, the DATA column in the table below
contains measurements for
FIELD 0 with 4096 channels and 4 correlations
*and* measurements for
FIELD 1 with 64 channels and 2 correlations.

+-----------------------------------------------------------------------+
|                                  MAIN                                 |
+-----+----------+--------------+------+----------+----------+----------+
| row | FIELD_ID | DATA_DESC_ID | TIME | ANTENNA1 | ANTENNA2 | DATA     |
+-----+----------+--------------+------+----------+----------+----------+
| 0   | 0        | 0            | 0.1  | 0        | 1        | (4096,4) |
+-----+----------+--------------+------+----------+----------+----------+
| 1   | 0        | 0            | 0.1  | 0        | 2        | (4096,4) |
+-----+----------+--------------+------+----------+----------+----------+
| 2   | 0        | 0            | 0.1  | 0        | 3        | (4096,4) |
+-----+----------+--------------+------+----------+----------+----------+
| 6   | 1        | 1            | 0.5  | 0        | 1        | (64,2)   |
+-----+----------+--------------+------+----------+----------+----------+
| 7   | 1        | 1            | 0.5  | 0        | 1        | (64,2)   |
+-----+----------+--------------+------+----------+----------+----------+
| 8   | 1        | 1            | 0.5  | 0        | 1        | (64,2)   |
+-----+----------+--------------+------+----------+----------+----------+

Variability in column shape can be challenging to deal with,
especially when we wish to store variably shaped data in
fixed shape structures like `numpy <https://www.numpy.org/devdocs/user/>`_
or `dask <https://docs.dask.org/en/latest/array.html>`_ arrays.

Table designers usually provide mechanisms for those who ingest
the data to infer this shape information. The keen-eyed will notice
the introduction of a *DATA_DESC_ID* column, which will be used to
describe the row shapes. But first we need to understand how tables
can be related to one another.


Table Relations
~~~~~~~~~~~~~~~

Other tables can be related to our observational data. Consider
the FIELD table:

+----------------------------+
|            FIELD           |
+-----+----------+-----------+
| row | NAME     | PHASE_DIR |
+-----+----------+-----------+
| 0   | 3C147    | [30, 60]  |
+-----+----------+-----------+
| 1   | PKS-1934 | [45, 70]  |
+-----+----------+-----------+

The FIELD_ID from the MAIN table can be used to look up the appropriate row
in the FIELD table, and by implication the NAME and PHASE_DIR of the
observed field for the related measurement.

Similarly the ANTENNA1 and ANTENNA2 columns in the MAIN table can be
related to information in an ANTENNA table:

+--------------------------------+
|             ANTENNA            |
+-----+-----------+--------------+
| row | NAME      | POSITION     |
+-----+-----------+--------------+
| 0   | STATION-1 | [10, 20, 30] |
+-----+-----------+--------------+
| 1   | STATION-2 | [15, 25, 35] |
+-----+-----------+--------------+
| 2   | STATION-3 | [20, 30, 40] |
+-----+-----------+--------------+
| 3   | STATION-4 | [25, 35, 45] |
+-----+-----------+--------------+


Returning to our variably shaped DATA problem described in
`Column Types and Shapes`_ we can define multiple DATA_DESCRIPTOR's,
referenced by the DATA_DESC_ID column in the MAIN table:

+---------------------------+
|      DATA_DESCRIPTOR      |
+-----+----------+----------+
| row | NUM_CHAN | NUM_CORR |
+-----+----------+----------+
| 0   | 4096     | 4        |
+-----+----------+----------+
| 1   | 64       | 2        |
+-----+----------+----------+

Each row defines the number of channels and correlations
for the DATA column in the MAIN table *via the DATA_DESC_ID* column.
Put another way, if we know the DATA_DESC_ID for each row,
then we can also know the number of channels and correlations for each row.


Measurement Sets
~~~~~~~~~~~~~~~~

A Measurement Set is a a collection of CASA Tables designed to store
observational data produced by a Radio Interferometer. The directory
structure for a WSRT.MS looks as follows:

.. code-block:: bash

    $ tree WSRT.MS
    WSRT.MS
    ├── ANTENNA
    │   ├── table.dat
    │   ├── table.f0
    │   ├── table.info
    │   └── table.lock
    ├── FIELD
    │   ├── table.dat
    │   ├── table.f0
    │   ├── table.f0i
    │   ├── table.info
    │   └── table.lock
    | ...
    ├── table.dat
    ├── table.f0
    ├── table.f0i
    ├── table.f1
    ├── table.f2
    ├── table.f2_TSM0
    | ...
    ├── table.info
    └── table.lock


`WSRT.MS` is a directory containing table data. Within this directory,
associated sub-tables are stored in the ANTENNA and FIELD sub-directories,
for example.

The example tables in this section are simplified versions
of the MAIN table and its sub-tables. The full structure of the
Measurement Set and it's sub-tables
is defined in the `Measurement Set v2.0 Specification
<https://casacore.github.io/casacore-notes/229.html>`_.

