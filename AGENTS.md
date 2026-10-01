# dask-ms: xarray Datasets from CASA Tables

[ratt-ru/dask-ms](https://github.com/ratt-ru/dask-ms) constructs xarray
`Datasets` from CASA Tables, such as the
[Measurement Set v2](https://casa.nrao.edu/Memos/229.html).
The `Variables` in each `Dataset` are dask arrays backed by deferred reads
through [arcae](https://github.com/ratt-ru/arcae),
which release the GIL and read through multiple independent table instances.
`Variables` can be written back to their respective Table columns.

### Core API

Read a Measurement Set as a list of lazy Datasets, one per partition
(`FIELD_ID`, `DATA_DESC_ID` by default), modify them and write them back:

```python
import dask
from daskms import xds_from_ms, xds_to_table

datasets = xds_from_ms("/data/data.ms", columns=["DATA", "FLAG"])
datasets = [ds.assign(DATA=(ds.DATA.dims, ds.DATA.data * 2)) for ds in datasets]
writes = xds_to_table(datasets, "/data/data.ms", columns=["DATA"])
dask.compute(writes)
```

Other entry points in [daskms/__init__.py](daskms/__init__.py):

- `xds_from_table` / `xds_to_table` — arbitrary CASA tables, including
  sub-tables via `"table.ms::FIELD"`.
- `xds_from_storage_ms` / `xds_from_storage_table` / `xds_to_storage_table` —
  dispatch on the store type: CASA tables, or the zarr and parquet stores in
  [daskms/experimental](daskms/experimental).
- `Dataset` / `Variable` — a reduced, xarray-like Dataset used when xarray
  is not installed.

## Tooling

This package is managed by the `uv` tool. Dependencies are declared as
`[project.optional-dependencies]` extras, not dependency groups, so select them
with `--extra`.

- Install: `uv sync --extra testing --extra dev`.
  Add `--extra complete` for the arrow, zarr, s3, xarray and katdal backends,
  as CI does.
- Test: `uv run --extra testing py.test -s -vvv daskms/`.
  - `--applications` enables the `dask-ms` / `fragments` CLI tests,
    `--optional` and `--stress` enable optional and long running tests.
  - Tests requiring missing optional dependencies (pyarrow, zarr, xarray,
    katdal, s3fs) are silently skipped: check the skip count.
  - python-casacore is a test-only dependency, used as an independent oracle
    to verify what arcae wrote.
- Pre-commit hooks gate commits; run them before proposing a change as done:
  `uv run --extra dev pre-commit run -a`.
  Format with the pre-commit's pinned ruff (v0.1.3), not the `ruff` in the
  `dev` extra: newer ruff versions reformat untouched code.
- Docs: Sphinx; source in [docs/](docs), hosted on ReadTheDocs
  ([readthedocs.yml](readthedocs.yml)). Build with
  `uv run --extra docs sphinx-build docs docs/_build/html`.
- [Changelog](HISTORY.rst): provide entries under the `X.Y.Z (YYYY-MM-DD)`
  heading when creating a PR, referencing it with `` (:pr:`NNN`) ``.
- Releases: `tbump <version>` bumps `pyproject.toml` and
  `daskms/__init__.py`, commits `Bump to <version>`, and pushes a tag of the
  bare version. CI publishes tagged pushes to PyPI.
