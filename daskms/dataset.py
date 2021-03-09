# -*- coding: utf-8 -*-

try:
    from collections.abc import Mapping  # python 3.8
except ImportError:
    from collections import Mapping

from collections import OrderedDict

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None


class DimensionInferenceError(ValueError):
    pass


class ChunkInferenceError(ValueError):
    pass


def data_var_dims(data_vars):
    """ Returns a {dim: size} dictionary constructed from `data_vars` """
    dims = {}

    for k, var in data_vars.items():
        for d, s in zip(var.dims, var.shape):
            if d in dims and not np.isnan(s) and s != dims[d]:
                raise DimensionInferenceError("Existing dimension size %s for "
                                              "dimension '%s' is inconsistent "
                                              "with same dimension %s of "
                                              "array %s" % (s, d, dims[d], k))

            dims[d] = s

    return dims


def data_var_chunks(data_vars):
    """ Returns a {dim: chunks} dictionary constructed from `data_vars` """
    chunks = {}

    for k, var in data_vars.items():
        if not isinstance(var.data, da.Array):
            continue

        for d, c in zip(var.dims, var.chunks):
            if d in chunks and c != chunks[d]:
                raise ChunkInferenceError("Existing chunking %s for "
                                          "dimension '%s' is inconsistent "
                                          "with chunking %s for the "
                                          "same dimension of array %s" %
                                          (c, d, chunks[d], k))

            chunks[d] = c

    return chunks


def as_variable(args):
    if not isinstance(args, tuple):
        raise TypeError(f"args ({type(args)}) must be a tuple")

    try:
        return Variable(*args)
    except TypeError as e:
        if "takes at most" in str(e):
            raise TypeError("Invalid number of arguments in Variable tuple. "
                            "Must be a size 2 to 5 tuple of the form "
                            "(dims, array[, attrs[, encoding[, fastpath]]]) ")

        raise


def _convert_to_variable(k, v):
    """ Converts ``v`` to a :class:`daskms.dataset.Variable` """
    if isinstance(v, Variable):
        return v

    if xr and isinstance(v, (xr.DataArray, xr.Variable)):
        return as_variable((v.dims, v.data, v.attrs))

    if not isinstance(v, (tuple, list)):
        raise ValueError("'%s' must be a size 2 to 5 tuple of the form"
                         "(dims, array[, attrs[, encoding[, fastpath]]]) "
                         "tuple. Got '%s' instead," % (k, type(v)))

    return as_variable(v)


if xr is not None:
    from xarray import Dataset, Variable
else:
    # This class duplicates xarray's Frozen class in
    # https://github.com/pydata/xarray/blob/master/xarray/core/utils.py
    # See https://github.com/pydata/xarray/blob/master/LICENSE
    class Frozen(Mapping):
        """
        Wrapper around an object implementing the Mapping interface
        to make it immutable.
        """
        __slots__ = "mapping"

        def __init__(self, mapping):
            self.mapping = mapping

        def __getitem__(self, key):
            return self.mapping[key]

        def __iter__(self):
            return iter(self.mapping)

        def __len__(self):
            return len(self.mapping)

        def __contains__(self, key):
            return key in self.mapping

        def copy(self):
            return self.mapping.copy()

        def __repr__(self):
            return f"{type(self).__name__}({self.mapping})"

    class Variable:
        """
        Replicates a minimal subset of `xarray Variable
        <http://xarray.pydata.org/en/stable/generated/xarray.Variable.html>`_'s
        functionality.
        Exists to allows ``xarray`` to be an optional ``dask-ms`` dependency.
        """

        def __init__(self, dims, data, attrs=None):
            """
            Parameters
            ----------
            dims : str or tuple
                Dimension schema. e.g. :code:`('row', 'chan', 'corr')`
            data : :class:`numpy.ndarray` or :class:`dask.array.Array`
                Array
            attrs : dict or None
                Array metadata
            """
            if not isinstance(dims, (tuple, list)):
                dims = (dims,)

            self.dims = dims
            self.data = data
            self.attrs = attrs or {}

        @property
        def dtype(self):
            """ Array data type """
            return self.data.dtype

        @property
        def chunks(self):
            """ Array chunks if wrapping a dask array else None """
            if isinstance(self.data, da.Array):
                return self.data.chunks

            return None

        @property
        def values(self):
            """ Returns actual array values """
            if isinstance(self.data, da.Array):
                return self.data.compute()

            return self.data

        @property
        def shape(self):
            """ Array shape """
            return self.data.shape

        @property
        def ndim(self):
            """ Number of array dimensions """
            return self.data.ndim

        def __dask_keys__(self):
            return self.data.__dask_keys__()

        def __dask_graph__(self):
            if isinstance(self.data, da.Array):
                return self.data.__dask_graph__()

            return None

        def __dask_layers__(self):
            return self.data.__dask_layers__()

        @property
        def __dask_optimize__(self):
            return self.data.__dask_optimize__

        @property
        def __dask_scheduler__(self):
            return self.data.__dask_scheduler__

        @staticmethod
        def finalize_compute(results, fn, args, name, dims, attrs):
            return Variable(dims, fn(results, *args), attrs=attrs)

        def __dask_postcompute__(self):
            fn, args = self.data.__dask_postcompute__()

            name = (self.data.name if isinstance(self.data, da.Array)
                    else None)
            args = (fn, args, name, self.dims, self.attrs)
            return (self.finalize_compute, args)

        @staticmethod
        def finalize_persist(results, fn, args, name, dims, attrs):
            results = {k: v for k, v in results.items() if k[0] == name}
            return Variable(dims, fn(results, *args), attrs=attrs)

        def __dask_postpersist__(self):
            fn, args = self.data.__dask_postpersist__()
            args = (fn, args, self.data.name, self.dims, self.attrs)
            return (self.finalize_persist, args)

    class Dataset:
        """
        Replicates a minimal subset of `xarray Dataset
        <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html#xarray.Dataset>`_'s
        functionality.
        Exists to allows ``xarray`` to be an optional ``dask-ms`` dependency.
        """

        def __init__(self, data_vars, coords=None, attrs=None):
            """
            Parameters
            ----------
            data_vars: dict
                Dictionary of variables of the form
                :code:`{name: (dims, array [, attrs])}`. `attrs` can
                be optional.
            coords : dict, optional
                Dictionary of coordinates of the form
                :code:`{name: (dims, array [, attrs])}`. `attrs` can
                be optional.
            attrs : dict, optional
                Dictionary of Dataset attributes
            """
            self._data_vars = {k: _convert_to_variable(k, v)
                               for k, v in data_vars.items()}

            if coords is not None:
                self._coords = {k: _convert_to_variable(k, v)
                                for k, v in coords.items()}
            else:
                self._coords = {}

            self._attrs = attrs or {}

        @property
        def attrs(self):
            """ Dataset attributes """
            return Frozen(self._attrs)

        @property
        def dims(self):
            """ A :code:`{dim: size}` dictionary """
            return data_var_dims(self._data_vars)

        sizes = dims

        @property
        def chunks(self):
            """ A :code:`{dim: chunks}` dictionary """
            return data_var_chunks(self._data_vars)

        @property
        def data_vars(self):
            """ Dataset variables """
            return Frozen(self._data_vars)

        @property
        def coords(self):
            """ Dataset coordinates """
            return Frozen(self._coords)

        def compute(self, **kwargs):
            """
            Calls dask compute on the dask arrays in this Dataset,
            returning a new Dataset.

            Returns
            -------
            :class:`~daskms.dataset.Dataset`
                Dataset containing computed arrays.
            """

            # Compute dask arrays separately
            dask_data = {}
            data_vars = {}

            # Split variables into dask and other data
            for k, v in self._data_vars.items():
                if isinstance(v.data, da.Array):
                    dask_data[k] = v
                else:
                    data_vars[k] = v

            # Compute dask arrays if present and add them to data variables
            if len(dask_data) > 0:
                data_vars.update(da.compute(dask_data, **kwargs)[0])

            return Dataset(data_vars,
                           coords=self._coords,
                           attrs=self._attrs.copy())

        def assign(self, **kwargs):
            r"""
            Creates a new Dataset from existing variables combined with
            those supplied in \*\*kwargs.

            Returns
            -------
            :class:`~daskms.dataset.Dataset`
                Dataset containing existing variables combined with
                those in \*\*kwargs.
            """
            data_vars = self._data_vars.copy()

            for k, v in kwargs.items():
                data_vars[k] = v

            return Dataset(data_vars,
                           attrs=self._attrs.copy(),
                           coords=self._coords)

        def assign_coords(self, **kwargs):
            r"""
            Creates a new Dataset from existing attributes combined with
            those supplied in \*\*kwargs.

            Returns
            -------
            :class:`~daskms.dataset.Dataset`
                Dataset containing existing attributes combined with
                those in \*\*kwargs.
            """

            coords = {k: as_variable(v) for k, v in kwargs.items()}
            return Dataset(self._data_vars, attrs=self._attrs, coords=coords)

        def assign_attrs(self, **kwargs):
            r"""
            Creates a new Dataset from existing attributes combined with
            those supplied in \*\*kwargs.

            Returns
            -------
            :class:`~daskms.dataset.Dataset`
                Dataset containing existing attributes combined with
                those in \*\*kwargs.
            """

            attrs = self._attrs.copy()
            attrs.update(kwargs)
            return Dataset(self._data_vars, attrs=attrs, coords=self._coords)

        @staticmethod
        def _drop_internal(mapping, names, errors="raise"):
            if isinstance(names, (tuple, list, set)):
                names = set(names)
            else:
                names = set([names])

            if errors == "raise":
                mapping = mapping.copy()
                for n in names:
                    try:
                        del mapping[n]
                    except KeyError:
                        raise ValueError(f"{n} does not exist on Dataset")
            elif errors == "ignore":
                return {k: v for k, v in mapping.items() if k not in names}
            else:
                raise ValueError(f"errors '{errors}' not in "
                                 f"('raise', 'ignore')")

        def drop_vars(self, names, *, errors):
            """Drop variables from the Dataset

            Parameters
            ----------
            names : str or iterable of str
                Variable names
            errors : {"raise", "ignore"}
                If "raise", a ValueError is raised if the
                specified variables are missing.
                If "ignore", the missing variables are ignored.

            Returns
            -------
            dataset : Dataset
                New dataset without the specified variables
            """
            data_vars = self._drop_internal(self.data_vars, names, errors)
            coords = self._drop_internal(self.coords, names, errors)
            return Dataset(data_vars, coords=coords, attrs=self.attrs.copy())

        def __getattr__(self, name):
            try:
                return self._data_vars[name]
            except KeyError:
                pass

            try:
                return self._coords[name]
            except KeyError:
                pass

            try:
                return self._attrs[name]
            except KeyError:
                raise AttributeError(f"Invalid Attribute {name}")

        def copy(self):
            """ Returns a copy of the Dataset """
            return Dataset(self._data_vars,
                           coords=self._coords,
                           attrs=self._attrs.copy())

        def __dask_graph__(self):
            graphs = {k: v.__dask_graph__() for k, v in self.data_vars.items()}
            # Excise anything that is not a dask collection
            graphs = {k: v for k, v in graphs.items() if v is not None}

            if len(graphs) > 0:
                return HighLevelGraph.merge(*graphs.values())

            return None

        def __dask_keys__(self):
            return [v.__dask_keys__()
                    for v in self._data_vars.values()
                    if dask.is_dask_collection(v)]

        def __dask_layers__(self):
            return sum([v.__dask_layers__()
                        for v in self._data_vars.values()
                        if dask.is_dask_collection(v)], ())

        @property
        def __dask_optimize__(self):
            return da.Array.__dask_optimize__

        @property
        def __dask_scheduler__(self):
            return da.Array.__dask_scheduler__

        @staticmethod
        def finalize_compute(results, info, coords, attrs):
            data_vars = OrderedDict()
            rev_results = list(results[::-1])

            for (dask_collection, k, v) in info:
                if dask_collection:
                    fn, args = v
                    r = rev_results.pop()
                    data_vars[k] = fn(r, *args)
                else:
                    data_vars[k] = v

            return Dataset(data_vars, coords=coords, attrs=attrs)

        def __dask_postcompute__(self):
            data_info = [
                (True, k, v.__dask_postcompute__())
                if dask.is_dask_collection(v)
                else (False, k, v)
                for k, v in self._data_vars.items()
            ]
            return self.finalize_compute, (
                data_info,
                self._coords,
                self._attrs
            )

        @staticmethod
        def finalize_persist(graph, info, coords, attrs):
            data_vars = OrderedDict()

            for dask_collection, k, v in info:
                if dask_collection:
                    fn, args = v
                    data_vars[k] = fn(graph, *args)
                else:
                    data_vars[k] = v

            return Dataset(data_vars, coords=coords, attrs=attrs)

        def __dask_postpersist__(self):
            data_info = [
                (True, k, v.__dask_postpersist__())
                if dask.is_dask_collection(v)
                else (False, k, v)
                for k, v in self._data_vars.items()
            ]
            return self.finalize_persist, (
                data_info,
                self._coords,
                self._attrs)
