# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from xarrayms.columns import infer_casa_type
from xarrayms.descriptors.builder import register_descriptor_builder
from xarrayms.descriptors.ms import MSDescriptorBuilder


@register_descriptor_builder("ratt_ms")
class RATTMSDescriptorBuilder(MSDescriptorBuilder):
    def default_descriptor(self):
        desc = super(RATTMSDescriptorBuilder, self).default_descriptor()

        desc['BITFLAG'] = {
            '_c_order': True,
            'comment': 'BITFLAG Column',
            'dataManagerGroup': 'StandardStMan',
            'dataManagerType': 'StandardStMan',
            'keywords': {'UNIT': 'Jy'},
            'maxlen': 0,
            'ndim': 2,  # (chan, corr)
            'option': 0,
            # comment this out to force type setting below
            # 'valueType': 'BYTE'}
        }

        return desc

    def descriptor(self, variables, default_desc):
        # Override the default BITFLAG type
        try:
            bitflag = variables['BITFLAG']
        except KeyError:
            pass
        else:
            casa_type = infer_casa_type(bitflag[0].dtype)
            default_desc['BITFLAG']['valueType'] = casa_type

        return super(RATTMSDescriptorBuilder, self).descriptor(variables,
                                                               default_desc)

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
