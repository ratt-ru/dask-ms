# -*- coding: utf-8 -*-


from daskms.descriptors.builder import register_descriptor_builder
from daskms.descriptors.ms import MSDescriptorBuilder


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
