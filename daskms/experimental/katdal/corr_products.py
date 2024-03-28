# Creation of the correlation product index is derived from
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


from collections import namedtuple

import numpy as np

CPInfo = namedtuple("CPInfo", "ant1_index ant2_index ant1 ant2 cp_index")


def corrprod_index(dataset, pols_to_use, include_auto_corrs=False):
    """The correlator product index (with -1 representing missing indices)."""
    corrprod_to_index = {tuple(cp): n for n, cp in enumerate(dataset.corr_products)}

    # ==========================================
    # Generate per-baseline antenna pairs and
    # correlator product indices
    # ==========================================

    def _cp_index(a1, a2, pol):
        """Create correlator product index from antenna pair and pol."""
        a1 = a1.name + pol[0].lower()
        a2 = a2.name + pol[1].lower()
        return corrprod_to_index.get((a1, a2), -1)

    # Generate baseline antenna pairs
    auto_corrs = 0 if include_auto_corrs else 1
    ant1_index, ant2_index = np.triu_indices(len(dataset.ants), auto_corrs)
    ant1_index, ant2_index = (a.astype(np.int32) for a in (ant1_index, ant2_index))

    # Order as similarly to the input as possible, which gives better performance
    # in permute_baselines.
    bl_indices = list(zip(ant1_index, ant2_index))
    bl_indices.sort(
        key=lambda ants: _cp_index(
            dataset.ants[ants[0]], dataset.ants[ants[1]], pols_to_use[0]
        )
    )
    # Undo the zip
    ant1_index[:] = [bl[0] for bl in bl_indices]
    ant2_index[:] = [bl[1] for bl in bl_indices]
    ant1 = [dataset.ants[a1] for a1 in ant1_index]
    ant2 = [dataset.ants[a2] for a2 in ant2_index]

    # Create actual correlator product index
    cp_index = [_cp_index(a1, a2, p) for a1, a2 in zip(ant1, ant2) for p in pols_to_use]
    cp_index = np.array(cp_index, dtype=np.int32)
    cp_index = cp_index.reshape(-1, len(pols_to_use))

    return CPInfo(ant1_index, ant2_index, ant1, ant2, cp_index)
