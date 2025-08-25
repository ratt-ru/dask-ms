# The numba transposition code is derived from
# https://github.com/ska-sa/katdal/blob/v0.22/scripts/mvftoms.py
# under the following license
#
# ################################################################################
# Copyright (c) 2011-2023, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


import dask.array as da
import numpy as np

from numba import njit, literally
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError, RequireLiteralValue


JIT_OPTIONS = {"nogil": True, "cache": True}


@njit(**JIT_OPTIONS)
def transpose_core(in_data, cp_index, data_type, row):
    return transpose_impl(in_data, cp_index, literally(data_type), literally(row))


def transpose_impl(in_data, cp_index, data_type_literal, row_literal):
    raise NotImplementedError


@overload(transpose_impl, jit_options=JIT_OPTIONS, prefer_literal=True)
def nb_transpose(in_data, cp_index, data_type_literal, row_literal):
    if not isinstance(data_type_literal, types.StringLiteral):
        raise RequireLiteralValue(
            f"'data_type' {data_type_literal} must be a StringLiteral"
        )

    if not isinstance(row_literal, types.BooleanLiteral):
        raise RequireLiteralValue(f"'row' {row_literal} must be a BooleanLiteral")

    DATA_TYPE = data_type_literal.literal_value
    ROW_DIM = row_literal.literal_value

    if DATA_TYPE == "flags":
        GET_VALUE = lambda v: v != 0
        DEFAULT_VALUE = np.bool_(True)
    elif DATA_TYPE == "vis":
        GET_VALUE = lambda v: v
        DEFAULT_VALUE = in_data.dtype(0 + 0j)
    elif DATA_TYPE == "weights":
        GET_VALUE = lambda v: v
        DEFAULT_VALUE = in_data.dtype(0)
    else:
        raise TypingError(f"data_type {DATA_TYPE} is not supported")

    GET_VALUE = register_jitable(GET_VALUE)

    def impl(in_data, cp_index, data_type_literal, row_literal):
        n_time, n_chans, _ = in_data.shape
        n_bls, n_pol = cp_index.shape
        out_data = np.empty((n_time, n_bls, n_chans, n_pol), in_data.dtype)

        bstep = 128
        bblocks = (n_bls + bstep - 1) // bstep
        for t in range(n_time):
            for bblock in range(bblocks):  # numba.prange
                bstart = bblock * bstep
                bstop = min(n_bls, bstart + bstep)
                for c in range(n_chans):
                    for b in range(bstart, bstop):
                        for p in range(out_data.shape[3]):
                            idx = cp_index[b, p]
                            data = (
                                GET_VALUE(in_data[t, c, idx])
                                if idx >= 0
                                else DEFAULT_VALUE
                            )
                            out_data[t, b, c, p] = data

        if ROW_DIM:
            return out_data.reshape(n_time * n_bls, n_chans, n_pol)

        return out_data

    return impl


def transpose(data, cp_index, data_type, row=False):
    ntime, _, _ = data.shape
    t_chunks, _, _ = data.chunks
    nbl, ncorr = cp_index.shape

    if row:
        out_dims = ("row", "chan", "corr")
        new_axes = {"row": ntime * nbl, "corr": ncorr}
    else:
        out_dims = ("time", "bl", "chan", "corr")
        new_axes = {"bl": nbl, "corr": ncorr}

    output = da.blockwise(
        transpose_core,
        out_dims,
        data,
        ("time", "chan", "corrprod"),
        cp_index,
        None,
        data_type,
        None,
        row,
        None,
        concatenate=True,
        new_axes=new_axes,
        dtype=data.dtype,
    )

    if row:
        return output.rechunk({0: tuple(tc * nbl for tc in t_chunks)})

    return output
