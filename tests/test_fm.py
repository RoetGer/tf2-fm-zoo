import pytest
import tensorflow as tf

from fm_zoo.fm import FactorizationMachine


def test_fm():
    x = [[1, 2, 2], [0, 3, 2]]
    m = FactorizationMachine([3, 4, 5], 2)
    o = m(x)

    assert o.shape == (2, 1)