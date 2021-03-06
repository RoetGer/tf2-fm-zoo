import pytest
import tensorflow as tf

from fm_zoo.ffm import FieldAwareFactorizationMachine


def test_ffm():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = FieldAwareFactorizationMachine([3, 4, 5], 2)
    o = m(x)
    
    assert o.shape == (2, 1)