# -*- coding: utf-8 -*-

from pathlib import Path
import os


def table_exists(table):
    if isinstance(table, Path):
        table = str(table)

    table = table.replace("::", os.sep)

    return os.path.exists(table) and os.path.isdir(table)
