from pathlib import Path, PurePath

import fsspec

from daskms.utils import freeze


class DaskMSStore:
    def __init__(self, url, **storage_options):
        if isinstance(url, PurePath):
            url = str(url)

        bits = url.split("::", 1)

        if len(bits) == 1:
            url = bits[0]
            table = ""
        elif len(bits) == 2:
            url, table = bits

            if table == "MAIN":
                raise ValueError("MAIN is a reserved table name")
        else:
            raise RuntimeError(f"len(bits): {len(bits)} not in (1, 2)")

        self.map = fsspec.get_mapper(url, **storage_options)
        self.fs = self.map.fs
        self.storage_options = storage_options
        protocol, path = fsspec.core.split_protocol(url)
        self.protocol = "file" if protocol is None else protocol

        if table:
            self.canonical_path = f"{path}::{table}"
            self.full_path = f"{path}{self.fs.sep}{table}"
            self.table = table
        else:
            self.canonical_path = path
            self.full_path = path
            self.table = "MAIN"

    def type(self):
        """
        Returns
        -------
        type : {"casa", "zarr", "parquet"}
            Type of table at the specified path

        Raises
        ------
        TypeError
            If it was not possible to infer the type of dataset
        """
        # From shallowest to deepest recursion
        if self.exists("table.dat"):
            return "casa"
        else:
            for _, _, files in self.fs.walk(self.full_path):
                for f in files:
                    if f == ".zgroup":
                        return "zarr"
                    elif f.endswith(".parquet"):
                        return "parquet"

        raise ValueError(f"Unable to infer table type at {self.full_path}")

    @property
    def url(self):
        return f"{self.fs.unstrip_protocol(self.canonical_path)}"

    def subdirectories(self):
        return [d["name"] for d
                in self.fs.listdir(self.full_path, detail=True)
                if d["type"] == "directory"]

    def casa_path(self):
        if self.protocol != "file":
            raise ValueError(f"CASA Tables are incompatible with the "
                             f"{self.protocol} protocol")

        return self.canonical_path

    @staticmethod
    def from_reduce_args(url, storage_options):
        return DaskMSStore(url, **storage_options)

    @staticmethod
    def from_url_and_kw(url, kw):
        if opts := kw.pop("storage_options", None):
            pass
        else:
            import json
            from appdirs import AppDirs

            path = Path(AppDirs("dask-ms").user_config_dir)
            path = path / "storage_options.json"

            if not path.exists():
                opts = {}
            else:
                with open(path, "r") as f:
                    opts = json.loads(f.read())

        return DaskMSStore(url, **opts)

    def __eq__(self, other):
        return (isinstance(other, DaskMSStore) and
                self.url == other.url and
                self.storage_options == other.storage_options)

    def __hash__(self):
        return hash(
            freeze(
                (
                    self.url,
                    self.storage_options,
                )
            )
        )

    def __reduce__(self):
        return (DaskMSStore.from_reduce_args,
                (self.url, self.storage_options))

    def __getitem__(self, key):
        return self.map[key]

    def _extend_path(self, path=""):
        return (f"{self.full_path}{self.fs.sep}{path}"
                if self.full_path
                else self.full_path)

    def exists(self, path=""):
        return self.fs.exists(self._extend_path(path))

    def ls(self, path=""):
        path = self._extend_path(path)
        return list(self.fs.ls(path, detail=False))

    @staticmethod
    def _remove_prefix(s, prefix):
        return s[len(prefix):] if s.startswith(prefix) else s

    def rglob(self, pattern, **kwargs):
        sep = self.fs.sep
        globpath = f"{self.full_path}{sep}**{sep}{pattern}"
        paths = self.fs.glob(globpath, **kwargs)
        prefix = f"{self.full_path}{sep}"
        return (self._remove_prefix(p, prefix) for p in paths)

    def open(self, key, *args, **kwargs):
        path = self._extend_path(key)
        return self.fs.open(path, *args, **kwargs)

    def makedirs(self, key, *args, **kwargs):
        path = self._extend_path(key)
        return self.fs.makedirs(path, *args, **kwargs)

    def rm(self, path="", recursive=False, maxdepth=None):
        path = self._extend_path(path)
        self.fs.rm(path, recursive=recursive, maxdepth=maxdepth)

    def subtable_store(self, subtable):
        if subtable == "MAIN":
            return self

        return DaskMSStore(f"{self.url}::{subtable}", **self.storage_options)

    def __repr__(self):
        return f"DaskMSStore({self.url})"

    def __str__(self):
        return self.url
