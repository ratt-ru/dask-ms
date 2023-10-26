import abc
from functools import partial
from pathlib import Path

from daskms.patterns import lazy_import

CASA_INPUT_ONLY_ARGS = ("group_columns", "index_columns", "taql_where")


class TableFormat(abc.ABC):
    @abc.abstractproperty
    def version(self):
        raise NotImplementedError

    @abc.abstractproperty
    def subtables(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def reader(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def writer(self):
        raise NotImplementedError

    @staticmethod
    def from_store(store):
        typ = store.type()

        if typ == "casa":
            from daskms.table_proxy import TableProxy
            import casacore.tables as ct

            table_proxy = TableProxy(ct.table, store.root, readonly=True, ack=False)
            keywords = table_proxy.getkeywords().result()
            subtables = CasaFormat.find_subtables(keywords)

            try:
                version = str(keywords["MS_VERSION"])
            except KeyError:
                cls = CasaMainFormat
                version = "<unspecified>"
            else:
                cls = MeasurementSetFormat

            main_fmt = cls(version, subtables)

            if store.table:
                return main_fmt.subtable_format(store.table)

            return main_fmt

        elif typ == "zarr":
            subtables = ZarrFormat.find_subtables(store)
            return ZarrFormat("0.1", subtables)
        elif typ == "parquet":
            subtables = ParquetFormat.find_subtables(store)
            return ParquetFormat("0.1", subtables)
        else:
            raise ValueError(f"Unexpected table type {typ}")

    @staticmethod
    def from_type(typ, subtable=""):
        if typ == "ms":
            if subtable:
                return MSSubtableFormat("2.0", subtable)
            else:
                return MeasurementSetFormat("2.0", [])
        elif typ == "casa":
            if subtable:
                return CasaSubtableFormat("<unspecified>", subtable)
            else:
                return CasaMainFormat("<unspecified>", [])
        elif typ == "zarr":
            return ZarrFormat("0.1", [])
        elif typ == "parquet":
            return ParquetFormat("0.1", [])
        else:
            raise ValueError(f"Unexpected table type {typ}")


class BaseTableFormat(TableFormat):
    def __init__(self, version):
        self._version = version

    @property
    def version(self):
        return self._version

    def check_unused_kwargs(self, fn_name, **kwargs):
        if kwargs:
            raise NotImplementedError(
                f"The following kwargs: "
                f"{list(kwargs.keys())} "
                f"were not consumed by "
                f"{self.__class__.__name__}."
                f"{fn_name}(**kw)"
            )


class CasaFormat(BaseTableFormat):
    TABLE_PREFIX = "Table: "

    @classmethod
    def find_subtables(cls, keywords):
        return [k for k, v in keywords.items() if cls.is_subtable(v)]

    @classmethod
    def is_subtable(cls, keyword: str):
        if not isinstance(keyword, str):
            return False

        if not keyword.startswith(cls.TABLE_PREFIX):
            return False

        path = Path(keyword[len(cls.TABLE_PREFIX) :])
        return path.exists() and path.is_dir() and (path / "table.dat").exists()


class CasaMainFormat(CasaFormat):
    def __init__(self, version, subtables):
        super().__init__(version)
        self._subtables = subtables

    def subtable_format(self, subtable: str):
        if subtable not in self._subtables:
            raise ValueError(f"{subtable} is not a valid subtable")

        return CasaSubtableFormat(self.version, subtable)

    @property
    def subtables(self):
        return self._subtables

    def subtable_format(self, subtable):
        return CasaSubtableFormat(self.version, subtable)

    def __str__(self):
        return "casa"


class CasaSubtableFormat(CasaFormat):
    def __init__(self, version, subtable):
        super().__init__(version)
        self._subtable = subtable

    @property
    def subtables(self):
        return []

    def reader(self, **kw):
        self.check_unused_kwargs("CasaSubtableFormat.reader", **kw)
        from daskms import xds_from_table

        return xds_from_table

    def writer(self):
        from daskms import xds_to_table

        return xds_to_table


class MSSubtableFormat(CasaSubtableFormat):
    def writer(self):
        from daskms import xds_to_table
        from daskms.table_schemas import SUBTABLES

        if self._subtable in SUBTABLES:
            descriptor = f"mssubtable('{self._subtable}')"
        else:
            descriptor = None

        return partial(xds_to_table, descriptor=descriptor)


class MeasurementSetFormat(CasaMainFormat):
    def __init__(self, version, subtables):
        super().__init__(version, subtables)

    def __str__(self):
        return "MeasurementSet"

    def reader(self, **kw):
        group_cols = kw.pop("group_columns", None)
        index_cols = kw.pop("index_columns", None)
        taql_where = kw.pop("taql_where", "")
        self.check_unused_kwargs("reader", **kw)

        from daskms import xds_from_ms

        return partial(
            xds_from_ms,
            group_cols=group_cols,
            index_cols=index_cols,
            taql_where=taql_where,
        )

    def writer(self):
        from daskms import xds_to_table

        return partial(xds_to_table, descriptor="ms")


class ZarrFormat(BaseTableFormat):
    def __init__(self, version, subtables):
        super().__init__(version)
        self._subtables = subtables

    @classmethod
    def find_subtables(cls, store):
        paths = (p.relative_to(store.path) for p in map(Path, store.subdirectories()))

        return [
            str(p)
            for p in paths
            if p.stem != "MAIN" and store.exists(str(p / ".zgroup"))
        ]

    @property
    def subtables(self):
        return self._subtables

    def reader(self, **kw):
        for arg in CASA_INPUT_ONLY_ARGS:
            if kw.pop(arg, False):
                raise ValueError(f'"{arg}" is not supported for zarr inputs')

        self.check_unused_kwargs("reader", **kw)

        from daskms.experimental.zarr import xds_from_zarr

        return xds_from_zarr

    def writer(self):
        from daskms.experimental.zarr import xds_to_zarr

        return xds_to_zarr

    def __str__(self):
        return "zarr"


class ParquetFormat(BaseTableFormat):
    def __init__(self, version, subtables):
        super().__init__(version)
        self._subtables = subtables

    @classmethod
    def find_subtables(cls, store):
        paths = (p.relative_to(store.path) for p in map(Path, store.subdirectories()))

        return [
            str(p)
            for p in paths
            if p.stem != "MAIN" and store.exists(str(p / ".zgroup"))
        ]

    @property
    def subtables(self):
        return self._subtables

    def reader(self, **kw):
        for arg in CASA_INPUT_ONLY_ARGS:
            if kw.pop(arg, False):
                raise ValueError(f'"{arg}" is not supported for parquet inputs')

        self.check_unused_kwargs("reader", **kw)

        from daskms.experimental.arrow.reads import xds_from_parquet

        return xds_from_parquet

    def writer(self):
        from daskms.experimental.arrow.writes import xds_to_parquet

        return xds_to_parquet

    def __str__(self):
        return "parquet"
