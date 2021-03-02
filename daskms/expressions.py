import ast
import sys

import dask.array as da

operators = {
    ast.Mult: da.multiply,
    ast.Div: da.divide,
    ast.FloorDiv: da.floor_divide,
    ast.Add: da.add,
    ast.Sub: da.subtract,
    ast.USub: da.negative
}


class Visitor(ast.NodeTransformer):
    def __init__(self, dataset):
        self.dataset = dataset

    if sys.version_info[1] >= 8:
        def visit_Constant(self, node):
            return node.n
    else:
        def visit_Num(self, node):
            return node.n

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_UnaryOp(self, node):
        try:
            op = operators[type(node.op)]
        except KeyError:
            raise ValueError(f"Unsupported operator {type(node.op)}")

        return op(self.visit(node.operand))

    def visit_BinOp(self, node):
        try:
            op = operators[type(node.op)]
        except KeyError:
            raise ValueError(f"Unsupported operator {type(node.op)}")

        return op(self.visit(node.left), self.visit(node.right))

    def visit_Name(self, node):
        return getattr(self.dataset, node.id).data


class DataColumnParseError(SyntaxError):
    pass


def data_column_expr(statement, datasets):
    """
    Produces a list of dask arrays with a
    variable set to the result of the
    supplied expression:

    .. code-block:: python

        vis = data_column_expr("DATA / (DIR1_DATA + DIR2_DATA)",
                               datasets)
        flag(vis)

    Parameters
    ----------
    expression : str
        For example, :code:`DATA / (DIR1_DATA + DIR2_DATA + DIR3_DATA)`.
        Can contain data column names as well as numeric literal values.
    datasets : list of Datasets or Dataset
        Datasets containing the DATA columns referenced in the statement

    Returns
    -------
    arrays : :class:`dask.array.Array` or list of :class:`dask.array.Array`
        list of expression results
    """
    if isinstance(datasets, (list, tuple)):
        promoted_datasets = list(datasets)
    else:
        promoted_datasets = [datasets]

    mod = ast.parse(statement)
    assert isinstance(mod, ast.Module)

    if len(mod.body) != 1:
        raise ValueError("Single Expression Only")

    expr = mod.body[0]

    if not isinstance(expr, ast.Expr):
        raise ValueError("Single Expression Only")

    expressions = []

    for ds in promoted_datasets:
        try:
            expressions.append(Visitor(ds).visit(expr))
        except SyntaxError:
            raise DataColumnParseError(f"Error parsing '{expr}'")

    if isinstance(datasets, (list, tuple)):
        return expressions
    else:
        return expressions[0]
