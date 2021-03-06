import pytest
import tensorflow as tf

from fm_zoo.common import EmbedFeatures, FieldAwareEmbedFeatures, FullyConnectedNetwork, LinearModel


def test_linear_model():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = LinearModel([3, 4, 5])
    o = m(x)
    
    assert o.shape == (2, 1)


def test_feature_embedding():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = EmbedFeatures([3, 4, 5], 2)
    o = m(x)
    
    assert o.shape == (2, 3, 2)


def test_field_aware_feature_embedding():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = FieldAwareEmbedFeatures([3, 4, 5], 2)
    o = m(x)
    
    assert o.shape == (2, 3, 3, 2)


def test_fully_connected():
    x = tf.random.uniform((32, 400))
    m = FullyConnectedNetwork([256, 128, 64, 1], .1)
    o = m(x)
    
    assert o.shape == (32, 1)