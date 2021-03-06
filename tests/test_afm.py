import pytest
import tensorflow as tf

from fm_zoo.afm import AttentionalFactorizationMachine


def test_afm():
    x = tf.convert_to_tensor([[1, 2, 2, 1, 5], [0, 3, 2, 2, 4], [2, 2, 2, 1, 1]], dtype="int64")
    m = AttentionalFactorizationMachine([3, 4, 5, 4, 6], 4, 2)
    o = m(x)
    
    assert o.shape == (3, 1)