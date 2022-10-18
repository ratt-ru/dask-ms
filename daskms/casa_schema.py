import itertools
import logging
from pprint import pformat

from daskms.constants import CASA_KEYWORDS, DASKMS_METADATA
from daskms.columns import infer_casa_type
from daskms.dataset_schema import DatasetSchema, ColumnSchema, unify_schemas
from daskms.utils import Frozen

log = logging.getLogger(__name__)


class CasaUnificationError(ValueError):
    pass


class CasaSchema(DatasetSchema):
    @classmethod
    def unify_attrs(cls, attrs):
        if len(attrs) == 0:
            return {}

        metadata = set(map(Frozen, (a.get(DASKMS_METADATA, {}) for a in attrs)))

        if len(metadata) != 1:
            raise CasaUnificationError(f"{pformat(metadata)} are not consistent")

        return {DASKMS_METADATA: metadata.pop()}

    @classmethod
    def unify_column(cls, column, schema):
        it = zip(schema.dims, schema.shape, schema.attrs)
        new_dims = []
        new_shape = []
        new_attrs = []

        for _, (dims, shape, attrs) in enumerate(it):
            if len(dims) == 0:
                log.warning(
                    f"Ignoring column {column} with " f"zero-length dims {schema.dims}"
                )
                continue

            if dims[0] != "row":
                log.warning(
                    f"Ignoring column {column} without "
                    f"'row' as the starting dimension {schema.dims}"
                )
                continue

            new_dims.append(dims[1:])
            new_shape.append(shape[1:])
            new_attrs.append(Frozen(attrs))

        typ = set(schema.type)
        dtype = set(schema.dtype)
        dims = set(new_dims)
        shape = set(new_shape)
        attrs = set(new_attrs)

        if len(typ) != 1:
            raise CasaUnificationError(f"Inconsistent column types {typ}")

        if len(dtype) != 1:
            raise CasaUnificationError(f"Inconsistent column dtypes {dtype}")

        if len(dims) != 1:
            raise CasaUnificationError(f"Inconsistent column dims {dims}")

        if len(attrs) != 1:
            raise CasaUnificationError(f"Inconsistent column attributes {attrs}")

        chunks = None

        return ColumnSchema(
            typ.pop(), dims.pop(), dtype.pop(), chunks, shape, attrs.pop().value
        )

    @classmethod
    def from_datasets(cls, datasets):
        if not isinstance(datasets, (tuple, list)):
            datasets = [datasets]

        schemas = list(map(DatasetSchema.from_dataset, datasets))
        unified_schema = unify_schemas(schemas)

        data_vars = {
            c: cls.unify_column(c, s) for c, s in unified_schema.data_vars.items()
        }
        coords = {c: cls.unify_column(c, s) for c, s in unified_schema.coords.items()}
        attrs = cls.unify_attrs(unified_schema.attrs)

        return CasaSchema(data_vars, coords, attrs)

    def to_casa_schema(self):
        desc = {}

        variables = itertools.chain(self.data_vars.items(), self.coords.items())
        variables = filter(lambda x: x[0] != "ROWID", variables)

        for column, var in variables:
            casa_type = infer_casa_type(var.dtype)
            keywords = var.attrs.get(DASKMS_METADATA, {}).get(CASA_KEYWORDS, {})

            column_desc = {
                "_c_order": True,
                "comment": f"{column} column",
                "dataManagerGroup": "StandardStMan",
                "dataManagerType": "StandardStMan",
                "keywords": keywords,
                "maxlen": 0,
                "option": 0,
                "valueType": casa_type,
            }

            if var.ndim > 0:
                column_desc["ndim"] = var.ndim

            if len(var.shape) == 1:
                col_shape = var.shape.pop()

                if len(col_shape) > 0:
                    column_desc["shape"] = col_shape

            desc[column] = column_desc

        return desc
