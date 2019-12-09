# -*- coding: utf-8 -*-

import ast

from daskms.utils import table_path_split
from daskms.table_schemas import SUBTABLES


class ParseFunctionCallError(Exception):
    def __init__(self, fn_str):
        super(ParseFunctionCallError, self).__init__(
            "Expected a single function call composed of "
            "python literals. "
            "For example: \"fn(1, 'bob', 2.0, c=2, d='hello')\". "
            "Got %s" % fn_str)


def parse_function_call_string(fn_str):
    """
    Parses a "fn('1', 2, c='fred', d=2.0)" function call string.
    The args and kwarg values must be python literals.

    Returns
    -------
    fn : str
        Function name
    args : tuple
        Evaluated literals
    kwargs : dict
        Evaluated keyword literals
    """
    mod = ast.parse(fn_str)

    if not len(mod.body) == 1:
        raise ParseFunctionCallError(fn_str)

    expr = mod.body[0]

    if not isinstance(expr, ast.Expr):
        raise ParseFunctionCallError(fn_str)

    node = expr.value

    if isinstance(node, ast.Name):
        return node.id, (), {}
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ParseFunctionCallError(fn_str)

        try:
            args = tuple(ast.literal_eval(a) for a in node.args)
        except ValueError:
            raise ParseFunctionCallError(fn_str)

        try:
            kwargs = {kw.arg: ast.literal_eval(kw.value)
                      for kw in node.keywords}
        except ValueError:
            raise ParseFunctionCallError(fn_str)

        return node.func.id, args, kwargs

    raise ParseFunctionCallError(fn_str)


def filename_builder_factory(filename):
    """
    Returns a Table Descriptor Builder based on the filename.

    1. If ending with a '.ms' (case insensitive), its assumed
       a Measurement Set is being created.
    2. If ending in '::SUBTABLE' where SUBTABLE is a
       Measurement Set sub-table such as ANTENNA, SPECTRAL_WINDOW,
       its assumed that sub-table is being created.
    3. Otherwise its assumed a default CASA table is being created.


    Parameters
    ----------
    filename : str
        Table filename

    Returns
    -------
    builder : :class:`daskms.descriptors.builder.AbtractDescriptorBuilder`
        Table Descriptor builder based on the filename
    """
    _, table, subtable = table_path_split(filename)

    # Does this look like an MS
    if not subtable and table[-3:].upper().endswith('.MS'):
        from daskms.descriptors.ms import MSDescriptorBuilder
        return MSDescriptorBuilder()

    # Perhaps its an MS subtable?
    if subtable in SUBTABLES:
        from daskms.descriptors.ms_subtable import MSSubTableDescriptorBuilder
        return MSSubTableDescriptorBuilder(subtable)

    # Just a standard CASA Table I guess
    from daskms.descriptors.builder import DefaultDescriptorBuilder
    return DefaultDescriptorBuilder()


def string_builder_factory(builder_str):
    """
    Creates a Table Descriptor Builder based on builder_str.

    builder_str should be a string representing a function call
    composed entirely of python literals.

    .. code-block:: python

        "fn(1, 2.0, 'fred', c=4, d='bob')"

    ``fn`` should be a registered descriptor builder which will be
    called with the supplied *args and **kwargs.

    Parameters
    ----------
    builder_str : str
        Descriptor builder string.

    Returns
    -------
    builder : :class:`daskms.descriptors.builder.AbtractDescriptorBuilder`
        Table Descriptor builder based on the builder_str
    """

    import daskms.descriptors.register_default_builders  # noqa
    from daskms.descriptors.builder import descriptor_builders

    fn, args, kwargs = parse_function_call_string(builder_str)

    try:
        builder_cls = descriptor_builders[fn]
    except KeyError:
        pass
    else:
        return builder_cls(*args, **kwargs)

    raise ValueError("No builders registered for "
                     "builder string '%s'. "
                     "Perhaps the appropriate python module "
                     "registering the builder has not yet "
                     "been imported." % builder_str)
