import pytest
import tensorflow as tf

from fm_zoo.dfm import DeepFM


def test_dfm():
    x = [[1, 2, 2], [0, 3, 2]]
    m = DeepFM([3, 4, 5], 2, [5, 3, 1], .1)
    o = m(x)
    
    assert o.shape == (2, 1)