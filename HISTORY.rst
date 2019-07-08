=======
History
=======

0.10.0 (YYYY-MM-DD)
----------------------------

* Replace deprecated logging.warn with logging.warning (:pr:`37`)

0.1.9 (2019-06-21)
------------------

* Fix bug in generating unique tokens for table write operations (:pr:`36`)

0.1.8 (2019-06-19)
------------------

* Fix bug in tokenizing row runs (:pr:`35`).


0.1.7 (2019-06-19)
------------------

* Use dask HighLevelGraphs (:pr:`34`)

0.1.6 (2019-05-28)
------------------

* Fix python version package dependencies in universal wheels (:pr:`33`)

0.1.5 (2019-05-21)
------------------

* Restrict all CASA table access to a single thread per table (:pr:`31`)

0.1.4 (2019-05-03)
------------------

* Upgrade dask version and remove attr dependency (:pr:`28`)
* Fix user locking (:pr:`27`)
* Support TAQL WHERE clause (:pr:`25`)
* Support kwargs for pyrap.tables.table (:pr:`24`)
* Table schema fixes (:pr:`23`)

0.1.3 (2018-07-27)
------------------

* Introduce per-process table caching (:pr:`17`)

0.1.2 (2018-07-20)
------------------

* Mitigate fragmented row orderings (:pr:`12`)

0.1.1 (2018-06-01)
------------------

* Optimise getcol memory usage (:pr:`9`)

0.1.0 (2018-03-26)
------------------

* First release on PyPI.
