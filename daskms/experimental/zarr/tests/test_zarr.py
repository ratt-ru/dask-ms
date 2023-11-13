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
from daskms.fsspec_store import DaskMSStore, UnknownStoreTypeError

try:
    import xarray
except ImportError:
    xarray = None

try:
    import s3fs
except ImportError:
    s3fs = None


def test_zarr_string_array(tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("string-arrays") / "test.zarr"

    data = ["hello", "this", "strange new world", "full of", "interesting", "stuff"]
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


@pytest.mark.parametrize("consolidated", [True, False])
def test_metadata_consolidation(ms, ant_table, tmp_path_factory, consolidated):
    zarr_dir = tmp_path_factory.mktemp("zarr_store") / "test.zarr"
    ant_dir = zarr_dir.parent / f"{zarr_dir.name}::ANTENNA"

    main_store = DaskMSStore(zarr_dir)
    ant_store = DaskMSStore(ant_dir)

    ms_datasets = xds_from_ms(ms)
    ant_datasets = xds_from_table(ant_table)

    for ds in ms_datasets:
        ds.DATA.attrs["test-meta"] = {"payload": "foo"}

    for ds in ant_datasets:
        ds.POSITION.attrs["test-meta"] = {"payload": "foo"}

    main_store_writes = xds_to_zarr(ms_datasets, main_store, consolidated=consolidated)
    writes = [main_store_writes]
    writes.extend(xds_to_zarr(ant_datasets, ant_store, consolidated=consolidated))
    dask.compute(writes)

    assert main_store.exists("MAIN/.zmetadata") is consolidated
    assert ant_store.exists(".zmetadata") is consolidated

    if consolidated:
        with main_store.open("MAIN/.zmetadata") as f:
            assert "test-meta".encode("utf8") in f.read()

        with ant_store.open(".zmetadata") as f:
            assert "test-meta".encode("utf8") in f.read()

    for ds in xds_from_zarr(main_store, consolidated=consolidated):
        assert "test-meta" in ds.DATA.attrs

    for ds in xds_from_zarr(ant_store, consolidated=consolidated):
        assert "test-meta" in ds.POSITION.attrs


def zarr_tester(ms, spw_table, ant_table, zarr_store, spw_store, ant_store):
    ms_datasets = xds_from_ms(ms)
    spw_datasets = xds_from_table(spw_table, group_cols="__row__")
    ant_datasets = xds_from_table(ant_table)

    for i, ds in enumerate(ms_datasets):
        dims = ds.dims
        row, chan, corr = (dims[d] for d in ("row", "chan", "corr"))

        ms_datasets[i] = ds.assign_coords(
            **{
                "chan": (("chan",), np.arange(chan)),
                "corr": (("corr",), np.arange(corr)),
            }
        )

    main_zarr_writes = xds_to_zarr(
        ms_datasets, zarr_store.url, storage_options=zarr_store.storage_options
    )
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

    return zarr_tester(
        ms,
        spw_table,
        ant_table,
        DaskMSStore(zarr_store),
        DaskMSStore(spw_store),
        DaskMSStore(ant_store),
    )


@pytest.mark.skipif(s3fs is None, reason="s3fs not installed")
def test_xds_to_zarr_s3(
    ms, spw_table, ant_table, py_minio_client, minio_user_key, minio_url, s3_bucket_name
):
    py_minio_client.make_bucket(s3_bucket_name)

    zarr_store = DaskMSStore(
        f"s3://{s3_bucket_name}/measurementset.MS",
        key=minio_user_key,
        secret=minio_user_key,
        client_kwargs={
            "endpoint_url": minio_url,
            "region_name": "af-cpt",
        },
    )

    # NOTE(sjperkins)
    # Review this interface
    spw_store = zarr_store.subtable_store("SPECTRAL_WINDOW")
    ant_store = zarr_store.subtable_store("ANTENNA")

    return zarr_tester(ms, spw_table, ant_table, zarr_store, spw_store, ant_store)


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
            row=np.arange(row), chan=np.arange(chan), corr=np.arange(corr)
        )

    for i, ds in enumerate(datasets):
        ds.to_zarr(str(store / f"ds-{i}.zarr"))


def _fasteners_runner(lockfile):
    import fasteners
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
    pytest.importorskip("fasteners")
    lockfile = tmp_path_factory.mktemp("fasteners-") / "dir" / ".lock"

    from pprint import pprint

    with Pool(4) as pool:
        results = [pool.apply_async(_fasteners_runner, (lockfile,)) for _ in range(4)]
        pprint([r.get() for r in results])


def test_basic_roundtrip(tmp_path):
    path = tmp_path / "test.zarr"

    # We need >10 datasets to be sure roundtripping is consistent.
    xdsl = [Dataset({"x": (("y",), da.ones(i))}) for i in range(1, 12)]
    dask.compute(xds_to_zarr(xdsl, path))

    xdsl = xds_from_zarr(path)
    dask.compute(xds_to_zarr(xdsl, path))


@pytest.mark.skipif(xarray is None, reason="depends on xarray")
@pytest.mark.parametrize(
    "prechunking",
    [{"row": -1, "chan": -1}, {"row": 1, "chan": 1}, {"row": 2, "chan": 7}],
)
@pytest.mark.parametrize(
    "postchunking",
    [{"row": -1, "chan": -1}, {"row": 1, "chan": 1}, {"row": 2, "chan": 7}],
)
def test_rechunking(ms, tmp_path_factory, prechunking, postchunking):
    store = tmp_path_factory.mktemp("zarr_store")
    ref_datasets = xds_from_ms(ms)

    for i, ds in enumerate(ref_datasets):
        chunks = ds.chunks
        row = sum(chunks["row"])
        chan = sum(chunks["chan"])
        corr = sum(chunks["corr"])

        ref_datasets[i] = ds.assign_coords(
            row=np.arange(row),
            chan=np.arange(chan),
            corr=np.arange(corr),
            dummy=np.arange(10),  # Orphan coordinate.
        )

    chunked_datasets = [ds.chunk(prechunking) for ds in ref_datasets]
    dask.compute(xds_to_zarr(chunked_datasets, store))

    rechunked_datasets = [ds.chunk(postchunking) for ds in xds_from_zarr(store)]
    dask.compute(xds_to_zarr(rechunked_datasets, store, rechunk=True))

    rechunked_datasets = xds_from_zarr(store)

    assert all([ds.equals(rds) for ds, rds in zip(rechunked_datasets, ref_datasets)])


@pytest.mark.skipif(xarray is None, reason="depends on xarray")
@pytest.mark.parametrize(
    "prechunking",
    [{"row": -1, "chan": -1}, {"row": 1, "chan": 1}, {"row": 2, "chan": 7}],
)
@pytest.mark.parametrize(
    "postchunking",
    [{"row": -1, "chan": -1}, {"row": 1, "chan": 1}, {"row": 2, "chan": 7}],
)
def test_add_datavars(ms, tmp_path_factory, prechunking, postchunking):
    store = tmp_path_factory.mktemp("zarr_store")
    ref_datasets = xds_from_ms(ms)

    for i, ds in enumerate(ref_datasets):
        chunks = ds.chunks
        row = sum(chunks["row"])
        chan = sum(chunks["chan"])
        corr = sum(chunks["corr"])

        ref_datasets[i] = ds.assign_coords(
            row=np.arange(row),
            chan=np.arange(chan),
            corr=np.arange(corr),
            dummy=np.arange(10),  # Orphan coordinate.
        )

    chunked_datasets = [ds.chunk(prechunking) for ds in ref_datasets]
    dask.compute(xds_to_zarr(chunked_datasets, store))

    rechunked_datasets = [ds.chunk(postchunking) for ds in xds_from_zarr(store)]
    augmented_datasets = [
        ds.assign({"DUMMY": (("row", "chan", "corr"), da.zeros_like(ds.DATA.data))})
        for ds in rechunked_datasets
    ]
    dask.compute(xds_to_zarr(augmented_datasets, store, rechunk=True))

    augmented_datasets = xds_from_zarr(store)

    assert all(
        [
            ds.DUMMY.chunks == cds.DATA.chunks
            for ds, cds in zip(augmented_datasets, chunked_datasets)
        ]
    )


def test_zarr_2gb_limit(tmp_path_factory):
    store = tmp_path_factory.mktemp("zarr_store")

    chunk = (1024, 1024, 1024)
    datasets = Dataset(
        {"DATA": (("x", "y", "z"), da.zeros(chunk, chunks=chunk, dtype=np.uint16))}
    )

    with pytest.raises(ValueError, match="2GiB chunk limit"):
        xds_to_zarr(datasets, store)

    chunk = (1024, 1024, 999)
    datasets = Dataset(
        {"DATA": (("x", "y", "z"), da.zeros(chunk, chunks=chunk, dtype=np.uint16))}
    )

    xds_to_zarr(datasets, store)


def test_xds_from_zarr_assert_on_empty_store(tmp_path_factory, ms):
    path = tmp_path_factory.mktemp("zarr_store") / "test.zarr"

    with pytest.raises(UnknownStoreTypeError, match="Unable to infer table type"):
        xds_from_zarr(path)
