import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class SparseDeepHaloChoice(tf.keras.Model):
    """
    Combined Zhang (2025) DeepHalo and Lu (2025) Sparse Shocks.
    """

    def __init__(
            self,
            n_features,
            n_items,
            n_markets,
            n_layers=2,
            hidden_dim=32,
            shrinkage_scale=0.01,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.shrinkage_scale = shrinkage_scale

        # DeepHalo Components
        self.embedding_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])
        self.context_projections = [
            tf.keras.layers.Dense(hidden_dim) for _ in range(n_layers)
        ]
        self.modulators = [
            tf.keras.layers.Dense(hidden_dim, activation='tanh') for _ in range(n_layers)
        ]
        self.utility_head = tf.keras.layers.Dense(1)

        # Lu (2025) Sparse Shock Components
        self.xi = tf.Variable(
            tf.zeros(shape=(n_markets, n_items)),
            trainable=True
        )
        self.sparsity_prior = tfd.Laplace(loc=0.0, scale=self.shrinkage_scale)

    def _compute_utilities(self, X, market_ids, mask=None):
        # 1. DeepHalo Recursive Context Aggregation
        z_current = self.embedding_net(X)
        z0 = z_current

        mask_float = tf.cast(mask, tf.float32) if mask is not None else tf.ones_like(z0[:, :, :1])
        mask_expanded = tf.expand_dims(mask_float, -1)

        for l in range(self.n_layers):
            context_sum = tf.reduce_sum(z_current * mask_expanded, axis=1, keepdims=True)
            counts = tf.reduce_sum(mask_expanded, axis=1, keepdims=True) + 1e-8
            z_bar = context_sum / counts

            z_bar_projected = self.context_projections[l](z_bar)
            interaction = z_bar_projected * self.modulators[l](z0)
            z_current = z_current + interaction

        halo_utility = tf.squeeze(self.utility_head(z_current), axis=-1)

        # 2. Add Sparse Unobserved Shocks
        batch_xi = tf.gather(self.xi, market_ids)
        total_utility = halo_utility + batch_xi

        if mask is not None:
            very_neg = tf.constant(-1e9, dtype=total_utility.dtype)
            total_utility = tf.where(tf.cast(mask, tf.bool), total_utility, very_neg)

        return total_utility

    def train_step(self, data):
        # Unpack data including market IDs for the shocks
        X, market_ids, mask, choices = data

        with tf.GradientTape() as tape:
            logits = self._compute_utilities(X, market_ids, mask)

            nll_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=choices, logits=logits)
            )

            prior_loss = -tf.reduce_mean(self.sparsity_prior.log_prob(self.xi))
            total_loss = nll_loss + prior_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": total_loss, "nll_loss": nll_loss, "shrinkage_penalty": prior_loss}