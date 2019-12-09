=======
History
=======

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
