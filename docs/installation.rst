.. highlight:: shell

============
Installation
============


Stable release
--------------

To install dask-ms-2, run this command in your terminal:

.. code-block:: console

    $ pip install dask-ms-2

This is the preferred method to install dask-ms-2, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


casacure
--------

dask-ms-2 reads and writes CASA tables through casacure_, a pure-Rust
implementation of the CASA table system. It is installed with the package
and used automatically — there is nothing to select or configure.

casacure ships pre-built wheels for Linux, macOS and Windows, so there is
nothing to compile: no C/C++ casacore build, and no casacore Measures data to
download or keep up to date.

.. _casacure: https://github.com/tmolteno/casacure
