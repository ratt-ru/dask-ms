from ast import literal_eval
from numbers import Number
import operator
from pprint import pprint

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from moz_sql_parser import parse

class VisitError(ValueError):
    pass

# Temporary Implementation starts here
class SqlVisitor:
    op_map = {
        "eq": operator.eq,
        "neq": operator.ne,
        "lt": operator.lt,
        "lte": operator.le,
        "gt": operator.gt,
        "gte": operator.ge,
        "and": operator.and_,
        "or": operator.or_,
    }

    def __init__(self, statement, index):
        self.statement = statement
        self.index = index
        self.structure = parse(statement)

    def _visit_operator(self, node):
        if len(node) != 1:
            raise VisitError(f"{node} has multiple entries")

        op, operands = next(iter(node.items()))

        try:
            op = self.op_map[op]
        except KeyError as e:
            raise VisitError(f"{op} is not a recognised operator. "
                             f"{self.op_map.keys()}")

        return op(*(self._visit(o) for o in operands))

    def _visit_str(self, node):
        try:
            return self.index[node]
        except KeyError:
            pass

        try:
            return literal_eval(node)
        except Exception as e:
            raise VisitError(f"{node} is not an index array or a literal")

    def _visit(self, node):
        if isinstance(node, dict):
            return self._visit_operator(node)
        elif isinstance(node, list):
            return [self._visit(o) for o in node]
        elif isinstance(node, str):
            return self._visit_str(node)
        elif isinstance(node, Number):
            return node
        else:
            raise VisitError(f"Unhandled {node} with type {type(node)}")


    def visit(self):
        try:
            where = self.structure["where"]
        except KeyError:
            raise VisitError(f"{self.structure} doesn't contain a where clause")

        return self._visit(where)

# Test Cases start here
@pytest.fixture(params=[False])
def auto_corr(request):
    return request.param

@pytest.fixture(params=[3])
def ntime(request):
    return request.param

@pytest.fixture(params=[4])
def na(request):
    return request.param

@pytest.fixture
def time(ntime):
    return np.linspace(0.0, 10.0, ntime)

@pytest.fixture
def interval(ntime):
    return np.full(ntime, 1.0)

@pytest.fixture(params=[[0, 1]])
def ddid(request):
    return np.asarray(request.param)

@pytest.fixture(params=[[0, 1]])
def field(request):
    return np.asarray(request.param)

@pytest.fixture
def baselines(na, auto_corr):
    k = 0 if auto_corr else 1
    return tuple(map(np.int32, np.triu_indices(na, k)))

@pytest.fixture
def index(ddid, field, time, interval, baselines):
    ant1, ant2 = baselines

    # Setup to broadcast arrays against each other
    # (DDID, FIELD, TIME, BASELINE)
    d = {
        "DATA_DESC_ID": ddid[:, None, None, None],
        "FIELD_ID": field[None, :, None, None],
        "TIME": time[None, None, :, None],
        "INTERVAL": interval[None, None, :, None],
        "ANTENNA1": ant1[None, None, None, :],
        "ANTENNA2": ant2[None, None, None, :]

    }

    # Broadcast, ravel and create a dict
    arrays = (a.ravel() for a in np.broadcast_arrays(*d.values()))
    return dict(zip(d.keys(), arrays))

@pytest.mark.parametrize("statement", [
    "SELECT * FROM DATA WHERE (ANTENNA1 != ANTENNA2 AND DATA_DESC_ID = 0) OR FIELD_ID = 1",
    # "SELECT * FROM DATA WHERE ANTENNA1 = ANTENNA2",
    # "SELECT * FROM DATA WHERE ANTENNA1 > ANTENNA2",
    # "SELECT * FROM DATA WHERE ANTENNA1 < ANTENNA2",
    # "SELECT * FROM DATA WHERE ANTENNA1 <= ANTENNA2",
    # "SELECT * FROM DATA WHERE ANTENNA1 >= ANTENNA2",
])
@pytest.mark.parametrize("auto_corr", [True, False], indirect=True)
def test_selection(statement, index):
    visitor = SqlVisitor(statement, index)
    pprint(index)
    print(f"{index['TIME'].size} rows")
    mask = visitor.visit()

    expected = np.logical_or(
        np.logical_and(index["ANTENNA1"] != index["ANTENNA2"], index["DATA_DESC_ID"] == 0),
        index["FIELD_ID"] == 1)

    assert_array_equal(mask, expected)
