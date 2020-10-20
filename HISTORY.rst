=======
History
=======

0.2.6 (2020-10-20)
------------------
* Remove table close in ThreadPool for the last time (:pr:`122`)
* Respect the High Level Graph specification better during inline array creation (:pr:`123`)
* Support dictionary writes via putvarcol (:pr:`119`)
* Use getcell instead of getcellslice in sorted orderings (:pr:`120`)
* Update to pytest-flake8 1.0.6 (:pr:`117`)
* Test on Python 3.8 (:pr:`116`)
* Depend on python-casacore 3.3.1 (:pr:`116`)

0.2.5 (2020-05-11)
------------------
* Remove deadlock in TableProxy weakref.finalize on Python 3.6 (:pr:`113`)
* Use python-casacore wheels for travis testing, instead of kernsuite packages (:pr:`115`)

0.2.4 (2020-04-24)
------------------
* Documentation updates (:pr:`110`)
* Provide better warnings for unusual ROWID graphs during table updates (:pr:`108`)
* Work around casacore getcolslice caching (:pr:`107`)
* Update LICENSE year (:pr:`105`)
* Update license and production status in pypi classifiers (:pr:`104`)
* Use WHERE rather than HAVING clause in group ordering TAQL (:pr:`98`)
* Improve the dask task key names for clearer graph visualization (:pr:`102`)
* Cache and inline row runs in write operations (:pr:`96`)
* Support getcolslice and putcolslice in TableProxy (:pr:`91`)
* Use weakref.finalize to cleanup TableProxy and Executor objects (:pr:`89`)
* Pickle Executor key argument (:pr:`88`)
* Deprecate Python 3.5 support and test on Python 3.7 (:pr:`87`)
* Optionally expose TableProxy objects in dataset read/write methods (:pr:`85`)
* Upgrade to python-casacore 3.2 (:pr:`84`)
* Re-introduce xarray handling in dataset.as_variable (:pr:`83`)
* Explicitly require dask Arrays on write datasets (:pr:`83`)
* Document python-casacore install process (:pr:`80`, :pr:`81`)

0.2.3 (2019-12-09)
------------------
* Remove \_\_future\_\_ import (:pr:`79`)
* Update examples (:pr:`78`)
* Only log aggressively when the log level is DEBUG (:pr:`76`)
* Optimise dask graphs produced by dask-ms such that each data access node
  no longer has common ancestors but is instead an independent
  root node. This improves memory usage in case of the `predict
  <https://github.com/paoloserra/crystalball/issues/15#issuecomment-563170101>`_.
  (:pr:`75`)
* Read-lock TAQL row reference table by default (:pr:`74`)
* Produce write datasets rather a single concatenated dask array
  (:pr:`70`, :pr:`72`)


0.2.2 (2019-10-25)
------------------
* Fix spacing in TAQL WHERE queries (:pr:`68`)


0.2.1 (2019-10-23)
------------------

* Constrain table object access to Executor.
  Simplify table locking (:pr:`66`).
* Fix stress test (:pr:`65`)
* Remove keywords from variable attributes (:pr:`64`)

0.2.0 (2019-09-30)
------------------

* Fix and test non-standard sub-table creation (:pr:`60`)
* Improve sub-table creation logic (:pr:`59`, :pr:`60`)
* Support table and column keywords (:pr:`58`, :pr:`62`)
* Support concurrent access of multiple independent tables (:pr:`57`)
* Fix WEIGHT_SPECTRUM schema dimensions (:pr:`56`)
* Pin python-casacore to 3.0.0 (:pr:`54`)
* Drop python 2 support (:pr:`51`)
* Simplify Table Schemas (:pr:`50`)
* Add Concepts + Tutorial Documentation (:pr:`48`)
* Supporting reading and updating column keywords (:pr:`48`)
* Add OBSERVATION, FEED, POINTING, SOURCE table schemas (:pr:`48`)
* Remove single row squeezing in the `group_cols="__row__"` case (:pr:`48`)
* Handle multi-dimensional string arrays (:pr:`48`)
* Add preliminary example_ms (:pr:`48`)
* Add Concepts + Tutorial Documentation (:pr:`48`)
* Make xarray an optional dependency (:pr:`45`)
* Rename xarray-ms to dask-ms (:pr:`43`)
* Allow chunking by arbitrary dimensions (:pr:`41`)
* Add a simple Dataset, making xarray an optional dependency.
  (:pr:`41`, :pr:`46`, :pr:`47`, :pr:`52`)
* Add support for writing new tables from Datasets (:pr:`41`, :pr:`53`)
* Add support for appending to tables from Datasets (:pr:`41`)
