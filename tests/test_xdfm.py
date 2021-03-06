import pytest
import tensorflow as tf

from fm_zoo.xdfm import CompressedInteractionNetwork, ExtremeDeepFactorizationMachine


def test_cin():
    x = tf.random.uniform((32, 20, 5))
    cin = CompressedInteractionNetwork([100, 50, 10], split=True)
    o = cin(x)
    
    assert o.shape == (32, 1)


def test_xdfm():
    x = tf.convert_to_tensor([[1, 2, 2, 1, 5], [0, 3, 2, 2, 4], [2, 2, 2, 1, 1]])
    m = ExtremeDeepFactorizationMachine([3, 4, 5, 4, 6], 4, [20, 10, 1], [10, 5], .1, False)
    o = m(x)
    
    assert o.shape == (32, 1)