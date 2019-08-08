# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def table_descriptor(table):
    """
    Get the table descriptor
    """
    tabledesc = table._getdesc(actual=True)

    # Strip out 0 length "HCcoordnames" and "HCidnames"
    # as these aren't valid. (See tabledefinehypercolumn)
    for c, hcdef in tabledesc.get('_define_hypercolumn_', {}).items():
        if "HCcoordnames" in hcdef and len(hcdef["HCcoordnames"]) == 0:
            del hcdef["HCcoordnames"]
        if "HCidnames" in hcdef and len(hcdef["HCidnames"]) == 0:
            del hcdef["HCidnames"]

    return tabledesc
