import pytest
import tensorflow as tf

from fm_zoo.afi import AutomaticFeatureInteraction


def test_afi():
    x = tf.convert_to_tensor([[1, 2, 2, 1, 5], [0, 3, 2, 2, 4], [2, 2, 2, 1, 1]])
    m = AutomaticFeatureInteraction([3, 4, 5, 4, 6], 4, 2, 2, [5, 3, 1], .1)
    o = m(x)

    assert o.shape == (3, 1)