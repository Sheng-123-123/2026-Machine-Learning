import tensorflow as tf
import tensorflow_probability as tfp
from choice_learn.models.base_model import ChoiceModel

tfd = tfp.distributions


class SparseMarketShockChoice(ChoiceModel):
    """
    Scalable implementation of Lu (2025) using MAP estimation.
    Applies a Bayesian shrinkage prior to unobserved market shocks.
    """

    def __init__(self, n_items, n_markets, shrinkage_scale=0.01, **kwargs):
        super().__init__(**kwargs)
        self.n_items = n_items
        self.n_markets = n_markets

        # Prior scale controls the strictness of the sparsity assumption
        # Smaller scale = more shocks forced to strictly zero
        self.shrinkage_scale = shrinkage_scale

        # Mean preference weights for observable features
        self.beta = tf.keras.layers.Dense(1, use_bias=False, name="beta")

        # Unobserved sparse market-product shocks (xi)
        # Shape: (n_markets, n_items)
        self.xi = tf.Variable(
            tf.zeros(shape=(n_markets, n_items)),
            trainable=True,
            name="xi_sparse_shocks"
        )

        # TFP Shrinkage Prior (Laplace is mathematically equivalent to L1/LASSO)
        self.sparsity_prior = tfd.Laplace(loc=0.0, scale=self.shrinkage_scale)

    def compute_batch_utility(self, batch_features, batch_market_ids, mask=None):
        """
        batch_features: (batch_size, n_items, n_features)
        batch_market_ids: (batch_size,) indicating the market context
        """
        # Base utility from observable credit card features
        base_u = tf.squeeze(self.beta(batch_features), axis=-1)

        # Retrieve the unobserved shock for this specific market
        batch_xi = tf.gather(self.xi, batch_market_ids)

        # Aggregate utility (Base + Sparse Shock)
        utilities = base_u + batch_xi

        if mask is not None:
            mask_bool = tf.cast(mask, tf.bool)
            utilities = tf.where(mask_bool, utilities, tf.constant(-1e9, dtype=utilities.dtype))

        return utilities

    def train_step(self, data):
        # choice-learn pipelines typically unpack to features, availability, and choices
        features, market_ids, mask, choices = data

        with tf.GradientTape() as tape:
            logits = self.compute_batch_utility(features, market_ids, mask)

            # 1. Standard Multinomial Logit Cross-Entropy Loss
            nll_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=choices, logits=logits)
            )

            # 2. Bayesian Shrinkage Penalty (Negative Log Prior)
            # This forces the network to keep xi at zero unless the data strongly objects
            prior_loss = -tf.reduce_mean(self.sparsity_prior.log_prob(self.xi))

            # Total Loss (MAP objective)
            total_loss = nll_loss + prior_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": total_loss, "nll_loss": nll_loss, "shrinkage_penalty": prior_loss}