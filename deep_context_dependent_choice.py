import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DeepContextDependentChoice(tf.keras.Model):
    """
    Deep Context-Dependent Choice Model (pairwise version).

    Utility of alternative i in a choice set:
        U_i = f_base(x_i) + sum_{j != i} g_pair(x_i, x_j)

    where f_base and g_pair are neural networks shared across alternatives.
    """

    def __init__(
        self,
        n_features,
        base_hidden_units=(64, 64),
        pair_hidden_units=(64, 64),
        base_activation="relu",
        pair_activation="relu",
        l2_reg=0.0,
        name="deep_context_dependent_choice",
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of features per alternative.
        base_hidden_units : tuple of int
            Hidden layer sizes for base utility MLP f_base.
        pair_hidden_units : tuple of int
            Hidden layer sizes for pairwise context MLP g_pair.
        base_activation : str or callable
            Activation for f_base hidden layers.
        pair_activation : str or callable
            Activation for g_pair hidden layers.
        l2_reg : float
            L2 regularization strength for both networks.
        """
        super().__init__(name=name, **kwargs)

        self.n_features = n_features
        self.base_hidden_units = base_hidden_units
        self.pair_hidden_units = pair_hidden_units
        self.base_activation = base_activation
        self.pair_activation = pair_activation
        self.l2_reg = l2_reg

        kernel_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

        # f_base: intrinsic utility of each alternative from its own features
        base_layers = []
        for units in base_hidden_units:
            base_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=base_activation,
                    kernel_regularizer=kernel_reg,
                )
            )
        # Final scalar utility
        base_layers.append(
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_regularizer=kernel_reg,
            )
        )
        self.base_mlp = tf.keras.Sequential(base_layers, name="base_mlp")

        # g_pair: pairwise context influence of j on i, given (x_i, x_j)
        pair_layers = []
        for units in pair_hidden_units:
            pair_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=pair_activation,
                    kernel_regularizer=kernel_reg,
                )
            )
        # Final scalar context effect
        pair_layers.append(
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_regularizer=kernel_reg,
            )
        )
        self.pair_mlp = tf.keras.Sequential(pair_layers, name="pair_mlp")

    def _compute_utilities(self, X, mask=None, training=False):
        # X: (B, A, F)
        B = tf.shape(X)[0]
        A = tf.shape(X)[1]

        # 1. Base utility f_base(x_i)
        base_u = self.base_mlp(X, training=training)
        base_u = tf.squeeze(base_u, axis=-1)  # Result: (B, A)

        # 2. Pairwise Context
        Xi = tf.expand_dims(X, axis=2)  # (B, A, 1, F)
        Xj = tf.expand_dims(X, axis=1)  # (B, 1, A, F)

        # Leveraging broadcasting instead of explicit tf.tile saves memory
        Xi_broadcast = tf.broadcast_to(Xi, [B, A, A, self.n_features])
        Xj_broadcast = tf.broadcast_to(Xj, [B, A, A, self.n_features])

        pair_input = tf.concat([Xi_broadcast, Xj_broadcast], axis=-1)  # (B, A, A, 2F)
        pair_effect = self.pair_mlp(pair_input, training=training)
        pair_effect = tf.squeeze(pair_effect, axis=-1)  # (B, A, A)

        # 3. Apply Diagonal Mask (Self-influence is zero)
        diag_mask = 1.0 - tf.eye(A, dtype=pair_effect.dtype)
        pair_effect = pair_effect * diag_mask  # (B, A, A)

        # 4. Handle Optional Availability Mask
        if mask is not None:
            mask_f = tf.cast(mask, pair_effect.dtype)  # (B, A)
            # mask_j ensures unavailable alternatives don't affect others
            # mask_i ensures unavailable alternatives don't receive utility
            mask_2d = tf.expand_dims(mask_f, axis=1) * tf.expand_dims(mask_f, axis=2)
            pair_effect = pair_effect * mask_2d

        # 5. Aggregate and combine
        context_u = tf.reduce_sum(pair_effect, axis=-1)  # (B, A)
        utilities = base_u + context_u

        # 6. Final Logit Masking for Softmax
        if mask is not None:
            mask_bool = tf.cast(mask, tf.bool)
            very_neg = tf.constant(-1e9, dtype=utilities.dtype)
            utilities = tf.where(mask_bool, utilities, very_neg)

        return utilities
    # def _compute_utilities(self, X, mask=None, training=False):
    #     """
    #     Compute utilities (logits) for each alternative.
    #
    #     Parameters
    #     ----------
    #     X : tf.Tensor
    #         Shape (batch_size, n_alternatives, n_features)
    #     mask : tf.Tensor or None
    #         Optional availability mask of shape (batch_size, n_alternatives),
    #         where 1/True means available, 0/False means unavailable.
    #
    #     Returns
    #     -------
    #     logits : tf.Tensor
    #         Shape (batch_size, n_alternatives)
    #     """
    #     # X: (B, A, F)
    #     B = tf.shape(X)[0]
    #     A = tf.shape(X)[1]
    #
    #     # Base utility f_base(x_i) -> (B, A, 1) -> (B, A)
    #     base_u = self.base_mlp(X, training=training)
    #     base_u = tf.squeeze(base_u, axis=-1)  # (B, A)
    #
    #     # Pairwise context: for each ordered pair (i, j), build [x_i, x_j]
    #     # Xi: (B, A, 1, F), Xj: (B, 1, A, F)
    #     Xi = tf.expand_dims(X, axis=2)
    #     Xj = tf.expand_dims(X, axis=1)
    #
    #     # Tile to (B, A, A, F)
    #     Xi_tiled = tf.tile(Xi, [1, 1, A, 1])
    #     Xj_tiled = tf.tile(Xj, [1, A, 1, 1])
    #
    #     # Concatenate features [x_i, x_j]: (B, A, A, 2F)
    #     pair_input = tf.concat([Xi_tiled, Xj_tiled], axis=-1)
    #
    #     # g_pair(x_i, x_j) -> (B, A, A, 1) -> (B, A, A)
    #     pair_effect = self.pair_mlp(pair_input, training=training)
    #     pair_effect = tf.squeeze(pair_effect, axis=-1)  # (B, A, A)
    #
    #     # Remove self-effects: set diagonal elements g_pair(x_i, x_i) = 0
    #     zero_diag = tf.zeros([B, A], dtype=pair_effect.dtype)
    #     pair_effect = tf.linalg.set_diag(pair_effect, zero_diag)
    #
    #     # If mask is provided, ensure masked items do not influence others
    #     if mask is not None:
    #         mask_f = tf.cast(mask, pair_effect.dtype)  # (B, A)
    #         # Mask for i-dimension and j-dimension
    #         mask_i = tf.expand_dims(mask_f, axis=2)  # (B, A, 1)
    #         mask_j = tf.expand_dims(mask_f, axis=1)  # (B, 1, A)
    #         # Zero out effects where either i or j is unavailable
    #         pair_effect = pair_effect * mask_i * mask_j  # (B, A, A)
    #
    #     # Sum over j for each i: context from all other alternatives -> (B, A)
    #     context_u = tf.reduce_sum(pair_effect, axis=-1)
    #
    #     # Total utility
    #     utilities = base_u + context_u  # (B, A)
    #
    #     # Final masking for softmax: unavailable alts get huge negative utility
    #     if mask is not None:
    #         mask_bool = tf.cast(mask, tf.bool)
    #         very_neg = tf.constant(-1e9, dtype=utilities.dtype)
    #         utilities = tf.where(mask_bool, utilities, very_neg)
    #
    #     return utilities

    def call(self, inputs, training=False):
        """
        Forward pass to get choice probabilities.

        Parameters
        ----------
        inputs : tf.Tensor or (tf.Tensor, tf.Tensor)
            Either:
              - X of shape (batch_size, n_alternatives, n_features)
              - (X, mask) where mask shape is (batch_size, n_alternatives)

        Returns
        -------
        probs : tf.Tensor
            Shape (batch_size, n_alternatives)
        """
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            X, mask = inputs
        else:
            X, mask = inputs, None

        logits = self._compute_utilities(X, mask=mask, training=training)
        return tf.nn.softmax(logits, axis=-1)

    def log_prob(self, X, choices, mask=None, training=False):
        """
        Log-likelihood of observed choices.

        Parameters
        ----------
        X : tf.Tensor
            Shape (batch_size, n_alternatives, n_features)
        choices : tf.Tensor
            Shape (batch_size,), int indices of chosen alternatives.
        mask : tf.Tensor or None
            Availability mask (batch_size, n_alternatives).

        Returns
        -------
        log_prob : tf.Tensor
            Shape (batch_size,), log P(choice | X).
        """
        logits = self._compute_utilities(X, mask=mask, training=training)
        dist = tfd.Categorical(logits=logits)
        return dist.log_prob(choices)

    def neg_log_likelihood(self, X, choices, mask=None, sample_weight=None):
        """
        Convenience method if you want to use this directly with Keras losses.
        """
        log_lik = self.log_prob(X, choices, mask=mask, training=True)
        nll = -log_lik
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=nll.dtype)
            nll = nll * sample_weight
        return tf.reduce_mean(nll)


class DCDCWrapper(tf.keras.Model):
    """
    Keras wrapper around DeepContextDependentChoice with custom train_step.
    """

    def __init__(self, dcdc_model, **kwargs):
        super().__init__(**kwargs)
        self.dcdc = dcdc_model

    def call(self, inputs, training=False):
        """
        For inference: wrapper((X, mask)) or wrapper(X)
        """
        return self.dcdc(inputs, training=training)

    def train_step(self, data):
        # Keras passes (x, y) or (x, y, sample_weight)
        if len(data) == 2:
            x, y = data
            sample_weight = None
        elif len(data) == 3:
            x, y, sample_weight = data
        else:
            raise ValueError("train_step expects (x, y) or (x, y, sample_weight)")

        # x is either X or (X, mask)
        if isinstance(x, (tuple, list)) and len(x) == 2:
            X, mask = x
        else:
            X, mask = x, None

        with tf.GradientTape() as tape:
            log_prob = self.dcdc.log_prob(X, y, mask=mask, training=True)
            if sample_weight is None:
                loss = -tf.reduce_mean(log_prob)
            else:
                sample_weight = tf.cast(sample_weight, dtype=log_prob.dtype)
                loss = -tf.reduce_mean(log_prob * sample_weight)

            # Add regularization losses from Dense layers
            if self.dcdc.losses:
                loss += tf.add_n(self.dcdc.losses)

        gradients = tape.gradient(loss, self.dcdc.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dcdc.trainable_variables))
        return {"loss": loss}


if __name__ == "__main__":
    # Example usage / smoke test

    # Example shapes:
    # X_train: (N, A, F)
    # y_train: (N,)
    # mask_train: (N, A) or None
    N = 1000
    A = 10
    F = 20

    X_train = tf.random.normal(shape=(N, A, F))
    y_train = tf.random.uniform(shape=(N,), maxval=A, dtype=tf.int32)
    mask_train = tf.ones((N, A), dtype=tf.bool)  # all available in this toy example

    base_model = DeepContextDependentChoice(
        n_features=F,
        base_hidden_units=(64, 64),
        pair_hidden_units=(64, 64),
        l2_reg=1e-4,
    )

    wrapper = DCDCWrapper(base_model)
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    # IMPORTANT: pass mask as part of x, and labels as y
    wrapper.fit(
        x=(X_train, mask_train),  # (inputs, mask)
        y=y_train,                # labels: chosen alternative index
        epochs=10,
        batch_size=128,
    )

    # Inference example:
    probs = wrapper((X_train[:5], mask_train[:5]), training=False)
    print("Predicted choice probabilities (first 5 examples):")
    print(probs.numpy())
