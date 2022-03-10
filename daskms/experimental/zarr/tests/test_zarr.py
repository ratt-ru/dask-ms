import multiprocessing
from multiprocessing import Pool
import os

import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms import xds_from_ms, xds_from_table, xds_from_storage_ms
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.dataset import Dataset
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from daskms.fsspec_store import DaskMSStore

try:
    import xarray
except ImportError:
    xarray = None


def test_zarr_string_array(tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("string-arrays") / "test.zarr"

    data = ["hello", "this", "strange new world",
            "full of", "interesting", "stuff"]
    data = np.array(data, dtype=object).reshape(3, 2)
    data = da.from_array(data, chunks=((2, 1), (1, 1)))

    datasets = [Dataset({"DATA": (("x", "y"), data)})]
    writes = xds_to_zarr(datasets, zarr_store)
    dask.compute(writes)

    new_datasets = xds_from_zarr(zarr_store)

    assert len(new_datasets) == len(datasets)

    for nds, ds in zip(new_datasets, datasets):
        assert_array_equal(nds.DATA.data, ds.DATA.data)


def test_xds_to_zarr_coords(tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr_coords") / "test.zarr"

    data = da.ones((100, 16, 4), chunks=(10, 4, 1), dtype=np.complex64)
    rowid = da.arange(100, chunks=10)

    data_vars = {"DATA": (("row", "chan", "corr"), data)}
    coords = {
        "ROWID": (("row",), rowid),
        "chan": (("chan",), np.arange(16)),
        "foo": (("foo",), np.arange(4)),
    }

    ds = [Dataset(data_vars, coords=coords)]

    writes = xds_to_zarr(ds, zarr_store)
    dask.compute(writes)

    rds = xds_from_zarr(zarr_store)
    assert len(ds) == len(rds)

    for ods, nds in zip(ds, rds):
        for c, v in ods.data_vars.items():
            assert_array_equal(v.data, getattr(nds, c).data)

        for c, v in ods.coords.items():
            assert_array_equal(v.data, getattr(nds, c).data)


def zarr_tester(ms, spw_table, ant_table,
                zarr_store, spw_store, ant_store):

    ms_datasets = xds_from_ms(ms)
    spw_datasets = xds_from_table(spw_table, group_cols="__row__")
    ant_datasets = xds_from_table(ant_table)

    for i, ds in enumerate(ms_datasets):
        dims = ds.dims
        row, chan, corr = (dims[d] for d in ("row", "chan", "corr"))

        ms_datasets[i] = ds.assign_coords(**{
            "chan": (("chan",), np.arange(chan)),
            "corr": (("corr",), np.arange(corr)),
        })

    main_zarr_writes = xds_to_zarr(ms_datasets, zarr_store)
    assert len(ms_datasets) == len(main_zarr_writes)

    for ms_ds, zw_ds in zip(ms_datasets, main_zarr_writes):
        for k, _ in ms_ds.attrs[DASKMS_PARTITION_KEY]:
            assert getattr(ms_ds, k) == getattr(zw_ds, k)

    writes = [main_zarr_writes]
    writes.extend(xds_to_zarr(spw_datasets, spw_store))
    writes.extend(xds_to_zarr(ant_datasets, ant_store))
    dask.compute(writes)

    zarr_datasets = xds_from_storage_ms(zarr_store, chunks={"row": 1})

    for ms_ds, zarr_ds in zip(ms_datasets, zarr_datasets):
        # Check data variables
        assert ms_ds.data_vars, "MS Dataset has no variables"

        for name, var in ms_ds.data_vars.items():
            zdata = getattr(zarr_ds, name).data
            assert type(zdata) is type(var.data)  # noqa
            assert_array_equal(var.data, zdata)

        # Check coordinates
        assert ms_ds.coords, "MS Datset has no coordinates"

        for name, var in ms_ds.coords.items():
            zdata = getattr(zarr_ds, name).data
            assert type(zdata) is type(var.data)  # noqa
            assert_array_equal(var.data, zdata)

        # Check dataset attributes
        for k, v in ms_ds.attrs.items():
            zattr = getattr(zarr_ds, k)
            assert_array_equal(zattr, v)


def test_xds_to_zarr_local(ms, spw_table, ant_table, tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr_store") / "test.zarr"
    spw_store = zarr_store.parent / f"{zarr_store.name}::SPECTRAL_WINDOW"
    ant_store = zarr_store.parent / f"{zarr_store.name}::ANTENNA"

    return zarr_tester(ms, spw_table, ant_table,
                       zarr_store, spw_store, ant_store)


def test_xds_to_zarr_s3(ms, spw_table, ant_table,
                        py_minio_client, minio_user_key,
                        minio_url, s3_bucket_name):

    py_minio_client.make_bucket(s3_bucket_name)

    zarr_store = DaskMSStore(f"s3://{s3_bucket_name}/measurementset.MS",
                             key=minio_user_key,
                             secret=minio_user_key,
                             client_kwargs={
                                 "endpoint_url": minio_url.geturl(),
                                 "region_name": "af-cpt",
                             })

    # NOTE(sjperkins)
    # Review this interface
    spw_store = zarr_store.subtable_store("SPECTRAL_WINDOW")
    ant_store = zarr_store.subtable_store("ANTENNA")

    return zarr_tester(ms, spw_table, ant_table,
                       zarr_store, spw_store, ant_store)


@pytest.mark.skipif(xarray is None, reason="Needs xarray to rechunk")
def test_multiprocess_create(ms, tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr_store") / "test.zarr"

    ms_datasets = xds_from_ms(ms)

    for i, ds in enumerate(ms_datasets):
        ms_datasets[i] = ds.chunk({"row": 1})

    writes = xds_to_zarr(ms_datasets, zarr_store)

    ctx = multiprocessing.get_context("spawn")  # noqa
    dask.compute(writes, scheduler="processes")

    zds = xds_from_zarr(zarr_store)

    for zds, msds in zip(zds, ms_datasets):
        for k, v in msds.data_vars.items():
            assert_array_equal(v, getattr(zds, k))

        for k, v in msds.coords.items():
            assert_array_equal(v, getattr(zds, k))

        for k, v in msds.attrs.items():
            assert_array_equal(v, getattr(zds, k))


@pytest.mark.optional
@pytest.mark.skipif(xarray is None, reason="depends on xarray")
def test_xarray_to_zarr(ms, tmp_path_factory):
    store = tmp_path_factory.mktemp("zarr_store")
    datasets = xds_from_ms(ms)

    for i, ds in enumerate(datasets):
        chunks = ds.chunks
        row = sum(chunks["row"])
        chan = sum(chunks["chan"])
        corr = sum(chunks["corr"])

        datasets[i] = ds.assign_coords(
            row=np.arange(row),
            chan=np.arange(chan),
            corr=np.arange(corr))

    for i, ds in enumerate(datasets):
        ds.to_zarr(str(store / f"ds-{i}.zarr"))


def _fasteners_runner(lockfile):
    fasteners = pytest.importorskip("fasteners")
    from pathlib import Path
    import json

    lock = fasteners.InterProcessLock(lockfile)

    root = Path(lockfile).parents[0]
    metafile = root / "metadata.json"

    metadata = {"tables": ["MAIN", "SPECTRAL_WINDOW"]}

    with lock:
        if metafile.exists():
            exists = True

            with open(metafile, "r") as f:
                assert json.loads(f.read()) == metadata
        else:
            exists = False

            with open(metafile, "w") as f:
                f.write(json.dumps(metadata))

        return os.getpid(), exists


def test_fasteners(ms, tmp_path_factory):
    lockfile = tmp_path_factory.mktemp("fasteners-") / "dir" / ".lock"

    from pprint import pprint

    with Pool(4) as pool:
        results = [pool.apply_async(_fasteners_runner, (lockfile,))
                   for _ in range(4)]
        pprint([r.get() for r in results])


def test_basic_roundtrip(tmp_path):

    path = tmp_path / "test.zarr"

    # We need >10 datasets to be sure roundtripping is consistent.
    xdsl = [Dataset({'x': (('y',), da.ones(i))}) for i in range(1, 12)]
    dask.compute(xds_to_zarr(xdsl, path))

    xdsl = xds_from_zarr(path)
    dask.compute(xds_to_zarr(xdsl, path))
