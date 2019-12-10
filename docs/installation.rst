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


python-casacore
---------------

python-casacore is a `dependency <https://github.com/ska-sa/dask-ms/blob/83b09651f35b78b5e9f0ded3712bb7e10c496d1c/setup.py#L27_>`_
of dask-ms, used to access CASA tables. This means that when we do the following:


.. code-block:: console

    $ pip install dask-ms


pip will download python-casacore and try to install it.
There are binary wheels for versions of python-casacore >= 3.1.1 which,
in general, make the installation process trivial.

*However*, pip will attempt to build earlier versions from source.

Building python-casacore from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For pip to build python-casacore from source, the appropriate
C and C++ libraries must be installed otherwise this build process will fail.
The full list of packages are available here:

- https://github.com/casacore/casacore#requirements
- https://github.com/casacore/python-casacore#from-source

casacore Measures data
~~~~~~~~~~~~~~~~~~~~~~

Please complete me
