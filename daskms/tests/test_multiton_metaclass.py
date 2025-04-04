from functools import partial
import pickle
import time

from cacheout import Cache, LRUCache

from daskms.multiton import (
    MultitonMetaclass,
    FIVE_MINUTES,
    on_get_keep_alive,
    on_delete,
)

import pytest


def factory(a, b=2):
    """Test factory function"""
    return a + b


def test_default_multiton_cache():
    """Test default construction of the multiton cache"""

    class DefaultMultiton(metaclass=MultitonMetaclass):
        pass

    cache = DefaultMultiton._CACHE
    assert isinstance(cache, Cache)
    assert cache.ttl == FIVE_MINUTES
    assert cache.maxsize == 100
    assert isinstance(cache.on_get, partial)
    assert cache.on_get.func == on_get_keep_alive
    assert cache.on_get.args == (DefaultMultiton,)
    assert cache.on_delete == on_delete


def param_on_get(key, value, exist):
    pass


def param_on_delete(key, value, cause):
    pass


class ParameterisedMultiton(
    metaclass=MultitonMetaclass,
    cache_params={
        "cls": LRUCache,
        "ttl": 1,
        "maxsize": 10,
        "on_get": param_on_get,
        "on_delete": param_on_delete,
    },
):
    pass


def test_multiton_parameterisation():
    """Test parameterised construction of the multiton cache"""
    cache = ParameterisedMultiton._CACHE
    assert isinstance(cache, LRUCache)
    assert cache.ttl == 1.0
    assert cache.maxsize == 10
    assert cache.on_get == param_on_get
    assert cache.on_delete == param_on_delete


def test_multiton_ttl():
    """Tests that"""

    class TransientMultiton(
        metaclass=MultitonMetaclass, cache_params={"cls": LRUCache, "ttl": 0.01}
    ):
        """Multiton instances live for extremely short periods of time"""

        pass

    table = TransientMultiton(factory, 1, b=3)
    assert table.instance == 4
    assert len(TransientMultiton._CACHE) == 1
    time.sleep(0.0001)
    assert len(TransientMultiton._CACHE) == 1
    assert TransientMultiton._CACHE.get(table) is 4

    # Get refreshes the TTL
    for _ in range(3):
        time.sleep(0.009)
        assert len(TransientMultiton._CACHE) == 1
        assert TransientMultiton._CACHE.get(table) is 4

    # Get after the TTL exceeded fails
    time.sleep(0.01)
    assert len(TransientMultiton._CACHE) == 1
    assert TransientMultiton._CACHE.get(table) is None


class TableTestMultiton(metaclass=MultitonMetaclass):
    pass


@pytest.fixture
def clear_table_multiton_cache():
    yield
    TableTestMultiton._CACHE.clear()


def test_multiton_idempotency(clear_table_multiton_cache):
    """Test creating the same object with varying
    arg and kw combinations produce the same object"""
    m1 = TableTestMultiton(factory, 1, 3)
    m2 = TableTestMultiton(factory, 1, b=3)
    m3 = TableTestMultiton(factory, a=1, b=3)

    assert m1 is not m2 is not m3
    assert m1 == m2 == m3
    assert m1.instance is m2.instance is m3.instance

    assert set(TableTestMultiton._CACHE.keys()) == {m1}
    assert set(TableTestMultiton._CACHE.values()) == {4}


def test_multiton_independence(clear_table_multiton_cache):
    """Test that creating different objects with
    the same factory functions produces different cached
    objects"""
    m1 = TableTestMultiton(factory, 1, b=3)
    m2 = TableTestMultiton(factory, 2, b=3)

    assert m1 is not m2
    assert m1 != m2
    assert m1.instance != m2.instance

    assert set(TableTestMultiton._CACHE.items()) == {(m1, 4), (m2, 5)}


def test_multiton_release(clear_table_multiton_cache):
    """Test that releasing a multiton instance removes it
    from the cache"""
    m1 = TableTestMultiton(factory, 1, b=3)
    m1.instance  # Populate the cache
    assert set(TableTestMultiton._CACHE.items()) == {(m1, 4)}

    m1.release()

    assert len(TableTestMultiton._CACHE.items()) == 0


def test_multiton_serialisation(clear_table_multiton_cache):
    """Test multiton serialisation"""
    m1 = TableTestMultiton(factory, 1, b=3)
    m2 = pickle.loads(pickle.dumps(m1))

    assert m1 == m2
    assert m1 is not m2
    assert m1.instance == m2.instance
