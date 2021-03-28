import tensorflow as tf
from fm_zoo.common import LinearModel, EmbedFeatures


class FactorizationMachine(tf.keras.Model):
    """Implementation of Factorization Machines.

    Reference: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

    fm(x) = linear regression + pooling of degree d interactions between latent factors of dimension k.

    Degree d here is fixed at 2, so that we can simplify the calculation. Only hyperparam is factor_dim

    The paramer ´prior´ allows to set a prior distribution for the trained weights of the embeddings,
    this permits regularization of the weights.

    The input is discretized/Integer Encoded instead of real valued.
    See common.EmbedFeatures for detail on input configs.
   """
    def __init__(self, feature_cards, factor_dim, prior=None, name='factorization_machine'):
        super(FactorizationMachine, self).__init__(name=name)
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')
        
        if prior:
            self.prior = prior

    def call(self, x, training=False):
        linear_out = self.linear(x)
        factors = self.embedding(x)
        sum_of_squares = tf.reduce_sum(tf.pow(factors, 2), 1)
        square_of_sums = tf.pow(tf.reduce_sum(factors, 1), 2)
        interaction_out = 0.5 * tf.reduce_sum(square_of_sums - sum_of_squares, 1, keepdims=True)
        
        if self.prior:
            neg_prior_linear = - tf.reduce_sum(
                self.prior.log_prob(
                    self.linear.linear.embeddings)
            )
            neg_prior_embedd = - tf.reduce_sum(
                self.prior.log_prob(
                    self.embedding.embedding.embeddings)
            )

            self.add_loss(neg_prior_linear + neg_prior_embedd)
        
        return linear_out + interaction_out