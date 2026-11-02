.. highlight:: shell

============
Installation
============


Stable release
--------------

To install dask-ms-2 with its casacure_ table backend, run this command in
your terminal:

.. code-block:: console

    $ pip install "dask-ms-2[casacure]"

This is the preferred method to install dask-ms-2, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


casacure
--------

dask-ms-2 reads and writes CASA tables through casacure_, a pure-Rust
replacement for the casacore table system. The ``casacure`` extra installs
it:

.. code-block:: console

    $ pip install "dask-ms-2[casacure]"

casacure ships pre-built wheels for Linux, macOS and Windows, so there is
nothing to compile: no C/C++ casacore build, and no casacore Measures data to
download or keep up to date.

Selecting the backend
~~~~~~~~~~~~~~~~~~~~~

The backend is chosen with the ``DASK_MS_BACKEND`` environment variable,
which must be set before ``daskms`` is imported:

.. code-block:: console

    $ DASK_MS_BACKEND=casacure python my_pipeline.py

casacure is the supported backend of dask-ms-2. If ``DASK_MS_BACKEND`` is
unset, the package falls back to python-casacore
(``pip install "dask-ms-2[casacore]"``); that path exists for compatibility
only. Without either extra the package still imports, and raises when a table
is opened.

.. _casacure: https://github.com/tmolteno/casacure
