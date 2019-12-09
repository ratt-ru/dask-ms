# -*- coding: utf-8 -*-

import pyrap.tables as pt

from daskms.descriptors.builder import (register_descriptor_builder,
                                        AbstractDescriptorBuilder)
from daskms.table_schemas import SUBTABLES


@register_descriptor_builder("mssubtable")
class MSSubTableDescriptorBuilder(AbstractDescriptorBuilder):
    def __init__(self, subtable):
        if subtable not in SUBTABLES:
            raise ValueError("'%s' is not a valid Measurement Set "
                             "sub-table" % subtable)

        self.subtable = subtable
        self.DEFAULT_TABLE_DESC = pt.required_ms_desc(subtable)
        self.REQUIRED_FIELDS = set(self.DEFAULT_TABLE_DESC.keys())

    def default_descriptor(self):
        return self.DEFAULT_TABLE_DESC.copy()

    def descriptor(self, variables, default_desc):
        try:
            desc = {k: default_desc[k] for k in self.REQUIRED_FIELDS}
        except KeyError as e:
            raise RuntimeError("'%s' is not in REQUIRED_FIELDS" % str(e))

        # Now copy/create descriptors for supplied variables
        for k, v in variables.items():
            try:
                desc[k] = default_desc[k]
            except KeyError:
                desc[k] = self.variable_descriptor(k, v)

        return desc

    def dminfo(self, tab_desc):
        return {}
