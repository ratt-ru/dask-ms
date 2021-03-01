# -*- coding: utf-8 -*-

from pathlib import Path
import os


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


def table_exists(table):
    if isinstance(table, Path):
        table = str(table)

    table = table.replace("::", os.sep)

    return os.path.exists(table) and os.path.isdir(table)
