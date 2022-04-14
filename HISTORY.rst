=======
History
=======

X.Y.Z (YYYY-MM-DD)
------------------
* Fix #118 - raise more informative error. (:pr:`203`)
* Fix #201 - improve tiling. (:pr:`202`)
* Fix #199 - do not create spurious fields in zarr writes. (:pr:`200`)
* Improve fix for #172 - error out more reliably. (:pr:`198`)
* Fix #172 - error out when missing datavars should be written. (:pr:`197`)
* Fix #195 - allow non-standard columns to be tiled. (:pr:`196`)

0.2.8 (2022-04-06)
------------------
* Fix #176. Fix roundtripping of boolean tensor arrays. (:pr:`194`)
* Fix #175 for xds_from_storage_* functions. (:pr:`192`)
* Improve handling of subtables with variably sized rows in daskms-convert. (:pr:`191`)
* Ensure that `xds_from_zarr` sorts groups as integers and not strings (:pr:`188`)
* Ensure Natural Ordering for parquet files (:pr:`183)
* Fix xds_from_zarr and xds_from_parquet chunking behaviour (:pr:`182`)
* Add LazyProxy and LazyProxyMultiton patterns to dask-ms (:pr:`177`)
* Support cloud native storage formats via fsspec (:pr:`174`)


0.2.7 (2022-01-13)
------------------
* Fix inclusion of MANIFEST.in files (:pr:`173`)
* Add --group-columns to `dask-ms convert` for CASA Formats (:pr:`169`)
* Add ComplexArray -> numpy conversion (:pr:`168`)
* Ignore row dimension when fixing column shapes (:pr:`165`)
* Bump pip from 9.0.1 to 19.2 (:pr:`164`)
* Fix zarr coordinate writes (:pr:`162`)
* Deprecate Python 3.6 (:pr:`161`)
* Add IMAGING_WEIGHT_SPECTRUM to default Measurement Schema (:pr:`160`)
* Remove default time ordering from xds_from_ms (:pr:`156`)
* Make zarr writes completely lazy (:pr:`157`)
* Copy partitioning information when writing (:pr:`155`)
* Add a `dask-ms convert` script for converting between CASA, Zarr and Parquet formats (:pr:`145`)
* Convert code-base to f-strings with flynt (:pr:`144`)
* Consolidate Dataset Types into daskms.dataset (:pr:`143`)
* Correct Dataset persistence issues (:pr:`140`)
* Experimental arrow support (:pr:`130`, :pr:`132`, :pr:`133`, :pr:`135`, :pr:`136`, :pr:`138`, :pr:`145`)
* Experimental zarr support (:pr:`129`, :pr:`133`, :pr:`139`, :pr:`142`, :pr:`150`, :pr:`145`)
* Test data fix (:pr:`128`)
* Fix array inlining for writes (:pr:`126`)
* Allow Multi-Layer Inlining (:pr:`125`)
* Support DATA Column Expressions (:pr:`124`, :pr:`134`, :pr:`146`, :pr:`147`, :pr:`148`, :pr:`151`)


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
