# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin

import numpy as np
import pyrap.tables as pt

from daskms.reads import DatasetFactory


class TableChunking(object):
    def __init__(self, ms, group_cols=None, index_cols=None):
        self._ms = ms
        self._group_cols = group_cols
        self._index_cols = index_cols
        self._dataset_factory = None

    def __call__(self, *args, **kwargs):
        if self._dataset_factory is None:
            factory = DatasetFactory(self._ms, [],
                                     self._group_cols,
                                     self._index_cols)
            self._dataset_factory = factory
        else:
            factory = self._dataset_factory

        table_proxy = factory.table_proxy_factory()

        return self.chunk(table_proxy)

    def chunk(self, table_proxy):
        from pprint import pprint
        pprint(inspect_ms(self._ms))

        print(table_proxy)


class MSChunking(TableChunking):
    pass


# {(subtable, required): number_column}
SUBTABLES = {
    ("FEED", False): "NUM_RECEPTORS",
    ("FIELD", False): "NUM_POLY",
    ("POINTING", False): "NUM_POLY",
    ("POLARIZATION", True): "NUM_CORR",
    ("SOURCE", False): "NUM_LINES",
    ("SPECTRAL_WINDOW", True): "NUM_CHAN"
}


def _inspect_subtables(ms):
    for (subtable, required), num_column in SUBTABLES.items():
        subtable_path = pjoin(ms, subtable)
        subtable_name = "::".join((ms, subtable))

        if not os.path.isdir(subtable_path):
            if required:
                raise ValueError("%s required but not present", subtable)

            continue

        with pt.table(subtable_name, ack=False, readonly=True) as T:
            yield ((subtable, num_column), T.getcol(num_column))


def inspect_ms(ms):
    subtables = dict(_inspect_subtables(ms))

    ddid_name = "::".join((ms, "DATA_DESCRIPTION"))

    spw = subtables[("SPECTRAL_WINDOW", "NUM_CHAN")]
    pol = subtables[("POLARIZATION", "NUM_CORR")]

    with pt.table(ddid_name, ack=False, readonly=True) as T:
        spw_id = T.getcol("SPECTRAL_WINDOW_ID")
        pol_id = T.getcol("POLARIZATION_ID")

        nchan = spw[spw_id]
        ncorr = pol[pol_id]

        ddid_shapes = np.stack([nchan, ncorr], axis=1)
        return subtables, ddid_shapes






