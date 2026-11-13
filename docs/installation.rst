.. highlight:: shell

============
Installation
============


Stable release
--------------

To install dask-ms, run this command in your terminal:

.. code-block:: console

    $ pip install dask-ms

This is the preferred method to install dask-ms, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


arcae
-----

arcae is a `dependency <https://github.com/ska-sa/dask-ms/blob/master/pyproject.toml>`_
of dask-ms, used to access CASA tables. This means that when we do the
following:


.. code-block:: console

    $ pip install dask-ms


pip will download arcae and install it. arcae ships self-contained binary
wheels with casacore statically linked into the extension module, so there
are no C or C++ libraries to install first, and nothing is built from source.

dask-ms currently tracks an arcae pre-release. No ``--pre`` flag is needed:
the requirement names a pre-release version explicitly, which is enough for
pip to consider pre-releases for that package alone.

Updating casacore Measures data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arcae embeds casacore, whose Measurement system relates astronomical
objects to each other in space and time. Measures data is frequently updated
and casacore will complain if it is out of date.

The measures data can be downloaded at the location specified here:

- https://github.com/casacore/casacore#obtaining-measures-data

Uncompress the measures data to some appropriate location, such
as ``~/opt/casacore/data`` and point casacore at it by creating a
``.casarc`` file in your home directory with the following contents:

.. code-block:: ini

    measures.directory: ~/opt/casacore/data/
