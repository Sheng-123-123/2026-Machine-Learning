import tensorflow as tf

from deep_context_dependent_choice import DeepContextDependentChoice, DCDCWrapper

# Small toy data
# 2 observations (batch_size=2), 3 alternatives, 2 features per alternative
X_test = tf.constant(
    [
        [[1.0, 0.5], [0.2, -0.3], [0.0, 1.0]],   # obs 0
        [[-0.1, 0.8], [0.9, 0.1], [0.3, -0.4]]   # obs 1
    ],
    dtype=tf.float32
)  # shape (2, 3, 2)

# Mask: for obs 0 all 3 alts available; for obs 1 only first 2 available
mask_test = tf.constant(
    [
        [True, True, True],    # obs 0
        [True, True, False]    # obs 1: alt 2 unavailable
    ],
    dtype=tf.bool
)  # shape (2, 3)

# Suppose the chosen alternative indices are:
y_test = tf.constant([0, 1], dtype=tf.int32)  # obs 0 chose alt 0, obs 1 chose alt 1

# Build model
F = 2  # number of features
base_model = DeepContextDependentChoice(
    n_features=F,
    base_hidden_units=(32, 32),
    pair_hidden_units=(32, 32),
    l2_reg=1e-4,
)

wrapper = DCDCWrapper(base_model)
wrapper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# Forward pass: get choice probabilities on test data
probs = wrapper((X_test, mask_test), training=False)
print("Choice probabilities:\n", probs.numpy())

# Direct log-likelihood of y_test given X_test
log_lik = base_model.log_prob(X_test, y_test, mask=mask_test, training=False)
print("Log-likelihood per observation:\n", log_lik.numpy())
print("Average NLL:", -tf.reduce_mean(log_lik).numpy())




import tensorflow as tf

from deep_context_dependent_choice import DeepContextDependentChoice, DCDCWrapper  # adjust import


# -----------------
# 1. Create synthetic train/test data
# -----------------
N_train = 1000
N_test = 200
A = 5    # number of alternatives
F = 10   # number of features per alternative

# Random features
X_train = tf.random.normal(shape=(N_train, A, F))
X_test = tf.random.normal(shape=(N_test, A, F))

# All alternatives available (mask = all True)
mask_train = tf.ones((N_train, A), dtype=tf.bool)
mask_test = tf.ones((N_test, A), dtype=tf.bool)

# Random "true" probabilities for generating labels (toy model)
true_W = tf.random.normal(shape=(F, 1))
# Utilities: (N, A)
U_train = tf.tensordot(X_train, true_W, axes=[[2], [0]])  # (N, A, 1)
U_train = tf.squeeze(U_train, axis=-1)
U_test = tf.tensordot(X_test, true_W, axes=[[2], [0]])   # (N, A, 1)
U_test = tf.squeeze(U_test, axis=-1)

# Softmax to get probabilities
probs_train = tf.nn.softmax(U_train, axis=-1)  # (N, A)
probs_test = tf.nn.softmax(U_test, axis=-1)    # (N, A)

# Sample choices according to these probabilities
dist_train = tf.random.categorical(tf.math.log(probs_train), num_samples=1)
dist_test = tf.random.categorical(tf.math.log(probs_test), num_samples=1)

# Cast labels to int32 (important!)
y_train = tf.cast(tf.squeeze(dist_train, axis=-1), tf.int32)  # (N_train,)
y_test = tf.cast(tf.squeeze(dist_test, axis=-1), tf.int32)    # (N_test,)

# -----------------
# 2. Build and train your model
# -----------------
base_model = DeepContextDependentChoice(
    n_features=F,
    base_hidden_units=(64, 64),
    pair_hidden_units=(64, 64),
    l2_reg=1e-4,
)

wrapper = DCDCWrapper(base_model)
wrapper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# Train with fit
wrapper.fit(
    x=(X_train, mask_train),
    y=y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
)

# -----------------
# 3. Evaluate on test data
# -----------------
# Get predicted probabilities
probs_pred = wrapper((X_test, mask_test), training=False)  # (N_test, A)

# Predicted choice: argmax
y_pred = tf.argmax(probs_pred, axis=-1, output_type=tf.int32)

# Accuracy (now both int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), tf.float32))
print("Test accuracy:", accuracy.numpy())

# Average log-likelihood on test data
log_lik_test = base_model.log_prob(X_test, y_test, mask=mask_test, training=False)
avg_nll_test = -tf.reduce_mean(log_lik_test)
print("Test average NLL:", avg_nll_test.numpy())













