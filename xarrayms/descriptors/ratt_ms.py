# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from xarrayms.columns import infer_casa_type
from xarrayms.descriptors.builder import register_descriptor_builder
from xarrayms.descriptors.ms import MSDescriptorBuilder


@register_descriptor_builder("ratt_ms")
class RATTMSDescriptorBuilder(MSDescriptorBuilder):
    def fix_columns(self, variables, desc, dim_sizes):
        desc = super(RATTMSDescriptorBuilder, self).fix_columns(variables,
                                                                desc,
                                                                dim_sizes)

        try:
            chan = dim_sizes['chan']
            corr = dim_sizes['corr']
        except KeyError:
            pass
        else:
            self._maybe_fix_column('BITFLAG', desc, (chan, corr))

        return desc
