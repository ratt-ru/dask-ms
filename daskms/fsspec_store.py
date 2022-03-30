from pathlib import Path, PurePath

import fsspec

from daskms.patterns import Multiton


class DaskMSStore(metaclass=Multiton):
    def __init__(self, url, **storage_options):
        if isinstance(url, PurePath):
            url = str(url)

        self.url = url
        self.map = fsspec.get_mapper(url, **storage_options)
        self.fs = self.map.fs
        self.storage_options = storage_options
        protocol = fsspec.core.split_protocol(url)[0]
        self.protocol = "file" if protocol is None else protocol

        path = fsspec.core.strip_protocol(url)
        bits = path.split("::", 1)

        if len(bits) == 1:
            path = bits[0]
            table = "MAIN"
        elif len(bits) == 2:
            path, table = bits

            if table == "MAIN":
                raise ValueError("MAIN is a reserved table name")
        else:
            raise RuntimeError(f"len(bits): {len(bits)} not in (1, 2)")

        self.path = path
        self.table = table

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
            for _, _, files in self.fs.walk(self.path):
                for f in files:
                    if f == ".zgroup":
                        return "zarr"
                    elif f.endswith(".parquet"):
                        return "parquet"

        raise TypeError(f"Unknown table type at {self.url}")

    def subdirectories(self):
        return [d["name"] for d
                in self.fs.listdir(self.path, detail=True)
                if d["type"] == "directory"]

    def casa_path(self):
        if self.protocol != "file":
            raise ValueError(f"CASA Tables don't work with the "
                             f"{self.protocol} protocol")

        return (self.path if self.table == "MAIN"
                else f"{self.path}::{self.table}")

    @staticmethod
    def from_url_storage_options(url, storage_options):
        return DaskMSStore(url, **storage_options)

    def __reduce__(self):
        return (DaskMSStore.from_url_storage_options,
                (self.url, self.storage_options))

    def __getitem__(self, key):
        return self.map[key]

    def exists(self, path):
        fullpath = "".join((self.path, self.fs.sep, path))
        return self.fs.exists(fullpath)

    def ls(self, path=None):
        path = path or self.path
        return list(map(Path, self.fs.ls(path, detail=False)))

    @staticmethod
    def _remove_prefix(s, prefix):
        return s[len(prefix):] if s.startswith(prefix) else s

    def rglob(self, pattern, **kwargs):
        sep = self.fs.sep
        fullpath = "".join(
            (self.path, sep, self.table, sep, "**", sep, pattern)
        )
        paths = self.fs.glob(fullpath, **kwargs)
        prefix = "".join((self.path, sep))
        return (self._remove_prefix(p, prefix) for p in paths)

    def open_file(self, key, *args, **kwargs):
        fullpath = f"{self.path}{self.fs.sep}{key}"
        return self.fs.open(fullpath, *args, **kwargs)

    def makedirs(self, key, *args, **kwargs):
        fullpath = f"{self.path}{self.fs.sep}{key}"
        return self.fs.makedirs(fullpath, *args, **kwargs)

    def subtable_store(self, subtable):
        if subtable == "MAIN":
            return self

        fullpath = f"{self.url}{self.fs.sep}{subtable}"
        return DaskMSStore(fullpath, **self.storage_options)
