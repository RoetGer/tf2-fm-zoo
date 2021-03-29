import tensorflow as tf
from fm_zoo.common import LinearModel, EmbedFeatures


class AttentionalFactorizationMachine(tf.keras.Model):
    """Implementation of Attentional Factorization Machines.

    Reference: https://arxiv.org/abs/1708.04617

    It adds an attention network on the pairwise interactions of FM.
    """
    def __init__(self, feature_cards, factor_dim, attention_size, prior=None, name='afm'):
        super(AttentionalFactorizationMachine, self).__init__(name=name)
        self.num_features = len(feature_cards)
        self.factor_dim = factor_dim
        self.linear = LinearModel(feature_cards, prior=prior, name=name + '/linear_model')
        self.embedding = EmbedFeatures(feature_cards, factor_dim, prior=prior, name=name + '/feature_embedding')
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(units=attention_size, name=name + '/attention_hidden'),
            tf.keras.layers.ReLU(name=name + '/attention_activ'),
            tf.keras.layers.Dense(units=1, name=name + '/attention_logits'),
            tf.keras.layers.Softmax(axis=1, name=name + '/attention_score')
        ])

    def call(self, x, training=False):
        batch_size, num_features = int(tf.shape(x)[0]), self.num_features
        num_interactions = num_features * (num_features - 1) // 2

        factors = self.embedding(x)
        factors_i = tf.tile(tf.expand_dims(factors, 1), [1, num_features, 1, 1])
        factors_j = tf.tile(tf.expand_dims(factors, 2), [1, 1, num_features, 1])
        interactions = tf.multiply(factors_i, factors_j)

        # use a mask to get unique pairwise interactions
        mask = tf.ones(interactions.shape[:-1])
        mask = tf.cast(tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0), dtype=tf.bool)
        interactions = tf.boolean_mask(interactions, mask)

        interactions = tf.reshape(interactions, [batch_size, num_interactions, self.factor_dim])
        attention_scores = self.attention(interactions)
        interactions = tf.reduce_sum(interactions, axis=-1, keepdims=True)
        attended = tf.multiply(interactions, attention_scores)
        return self.linear(x) + tf.reduce_sum(attended, axis=1)
