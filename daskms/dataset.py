# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from collections.abc import Mapping  # python 3.8
except ImportError:
    from collections import Mapping

from collections import namedtuple

import dask
import dask.array as da


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

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.mapping)


class Variable(namedtuple("_Variable", ["dims", "data", "attrs"])):
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def chunks(self):
        if isinstance(self.data, da.Array):
            return self.data.chunks

        return None

    @property
    def values(self):
        if isinstance(self.data, da.Array):
            return self.data.compute()

        return self.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim


def data_var_dims(data_vars):
    """ Returns a {dim: size} dictionary constructed from `data_vars` """
    dims = {}

    for k, var in data_vars.items():
        for d, s in zip(var.dims, var.shape):

            if d in dims and s != dims[d]:
                raise ValueError("Existing dimension size %d for "
                                 "dimension '%s' is inconsistent "
                                 "with same dimension of array %s" %
                                 (s, d, k))

            dims[d] = s

    return dims


def data_var_chunks(data_vars):
    """ Returns a {dim: chunks} dictionary constructed from `data_vars` """
    chunks = {}

    for k, var in data_vars.items():
        if not isinstance(var.data, da.Array):
            continue

        for dim, c in zip(var.dims, var.chunks):
            if dim in chunks and c != chunks[dim]:
                raise ValueError("Existing chunking %s for "
                                 "dimension '%s' is inconsistent "
                                 "with chunking %s for the "
                                 "same dimension of array %s" %
                                 (c, dim, chunks[dim], k))

            chunks[dim] = c

    return chunks


def _convert_to_variable(k, v):
    """ Converts ``v`` to a :class:`daskms.dataset.Variable` """
    if isinstance(v, Variable):
        return v

    if not isinstance(v, (tuple, list)) and len(v) not in (2, 3):
        raise ValueError("'%s' must be a (dims, array) or "
                         "(dims, array, attrs) tuple. "
                         "Got '%s' instead," % (k, type(v)))

    dims = v[0]
    data = v[1]
    attrs = v[2] if len(v) > 2 else {}

    if len(dims) != data.ndim:
        raise ValueError("Dimension schema '%s' does "
                         "not match shape of associated array %s"
                         % (dims, data.shape))

    return Variable(dims, data, attrs)


class Dataset(object):
    """
    Poor man's xarray Dataset. It mostly exists so that xarray can
    be an optional dependency, as it in turn depends on pandas
    which is a fairly heavy dependency
    """

    def __init__(self, data_vars, attrs=None, coords=None):
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
        return Frozen(self._attrs)

    @property
    def dims(self):
        return data_var_dims(self._data_vars)

    sizes = dims

    @property
    def chunks(self):
        return data_var_chunks(self._data_vars)

    @property
    def data_vars(self):
        return Frozen(self._data_vars)

    @property
    def coords(self):
        return Frozen(self._coords)

    def compute(self):
        vnames, vdims, vdata, vattrs = zip(*tuple((k, v.dims, v.data, v.attrs)
                                                  for k, v
                                                  in self._data_vars.items()))

        cnames, cdims, cdata, cattrs = zip(*tuple((k, v.dims, v.data, v.attrs)
                                                  for k, v
                                                  in self._coords.items()))

        vdata, cdata = dask.compute(vdata, cdata)

        data_vars = {n: (d, v, a) for n, d, v, a
                     in zip(vnames, vdims, vdata, vattrs)}

        coords = {n: (d, v, a) for n, d, v, a
                  in zip(cnames, cdims, cdata, cattrs)}

        return Dataset(data_vars,
                       attrs=self._attrs.copy(),
                       coords=coords)

    def assign(self, **kwargs):
        data_vars = self._data_vars.copy()

        for k, v in kwargs.items():
            if not isinstance(v, (list, tuple)):
                try:
                    current_var = data_vars[k]
                except KeyError:
                    raise ValueError("Couldn't find existing dimension schema "
                                     "during assignment of variable '%s'. "
                                     "Supply a full (dims, array) tuple."
                                     % k)
                else:
                    data_vars[k] = (current_var.dims, v, current_var.attrs)
            else:
                data_vars[k] = v

        return Dataset(data_vars,
                       attrs=self._attrs.copy(),
                       coords=self._coords)

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
            raise AttributeError("Invalid Attribute %s" % name)

    def copy(self):
        return Dataset(self._data_vars,
                       attrs=self._attrs.copy(),
                       coords=self._coords)
