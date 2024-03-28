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
from numba.extending import overload, SentryLiteralArgs, register_jitable
from numba.core.errors import TypingError


JIT_OPTIONS = {"nogil": True, "cache": True}


@njit(**JIT_OPTIONS)
def transpose_core(in_data, cp_index, data_type, row):
    return transpose_impl(in_data, cp_index, data_type, row)


def transpose_impl(in_data, cp_index, data_type, row):
    raise NotImplementedError


@overload(transpose_impl, jit_options=JIT_OPTIONS, prefer_literal=True)
def nb_transpose(in_data, cp_index, data_type, row):
    SentryLiteralArgs(["data_type", "row"]).for_function(nb_transpose).bind(
        in_data, cp_index, data_type, row
    )

    try:
        data_type = data_type.literal_value
    except AttributeError as e:
        raise TypingError(f"data_type {data_type} is not a literal_value") from e
    else:
        if not isinstance(data_type, str):
            raise TypeError(f"data_type {data_type} is not a string: {type(data_type)}")

    try:
        row_dim = row.literal_value
    except AttributeError as e:
        raise TypingError(f"row {row} is not a literal_value") from e
    else:
        if not isinstance(row_dim, bool):
            raise TypingError(f"row_dim {row_dim} is not a boolean {type(row_dim)}")

    if data_type == "flags":
        get_value = lambda v: v != 0
        default_value = np.bool_(True)
    elif data_type == "vis":
        get_value = lambda v: v
        default_value = in_data.dtype(0 + 0j)
    elif data_type == "weights":
        get_value = lambda v: v
        default_value = in_data.dtype(0)
    else:
        raise TypingError(f"data_type {data_type} is not supported")

    get_value = register_jitable(get_value)

    def impl(in_data, cp_index, data_type, row):
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
                                get_value(in_data[t, c, idx])
                                if idx >= 0
                                else default_value
                            )
                            out_data[t, b, c, p] = data

        if row_dim:
            return out_data.reshape(n_time * n_bls, n_chans, n_pol)

        return out_data

    return impl


def transpose(data, cp_index, data_type, row=False):
    ntime, _, _ = data.shape
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
        literally(data_type),
        None,
        row,
        None,
        concatenate=True,
        new_axes=new_axes,
        dtype=data.dtype,
    )

    if row:
        return output.rechunk({0: ntime * (nbl,)})

    return output
