"""
Supervised autoencoder with optional class-weighted classification head.

Includes:
- Encoder: compresses input features to a latent space.
- Decoder: reconstructs input from latent space.
- Supervised head: predicts class labels from latent space.
- Handles class imbalance via weighted sparse categorical crossentropy.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils.class_weight import compute_class_weight

def supervised_autoencoder(input_dim, y_train,
                           latent_dim=32, units=[128, 64],
                           activation="relu", dropout=0.0,
                           learning_rate=0.001, optimizer="adam",
                           num_classes=7):
    """
    Build and compile a supervised autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    y_train : array-like
        Labels for supervised head to compute class weights.
    latent_dim : int
        Size of latent representation.
    units : list
        List of hidden units for encoder/decoder layers.
    activation : str
        Activation function for hidden layers.
    dropout : float
        Dropout rate for first hidden layer only.
    learning_rate : float
        Learning rate for optimizer.
    optimizer : str
        'adam' or 'sgd'.
    num_classes : int
        Number of classes for supervised head.

    Returns
    -------
    autoencoder : keras.Model
        Compiled autoencoder with reconstruction and classification outputs.
    encoder : keras.Model
        Encoder model to extract latent representations.
    """

    # ----- Encoder -----
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    dropout_cond = False
    for h in units:
        x = layers.Dense(h, activation=activation)(x)
        if dropout > 0 and not dropout_cond:
            x = layers.Dropout(dropout)(x)
            dropout_cond = True
    latent = layers.Dense(latent_dim, name="latent")(x)

    # ----- Decoder -----
    x = latent
    for h in reversed(units):
        x = layers.Dense(h, activation=activation)(x)
    reconstructed = layers.Dense(input_dim, name="reconstruction")(x)

    # ----- Supervised head -----
    supervised_out = layers.Dense(num_classes, activation="softmax", name="classification")(latent)

    # ----- Models -----
    autoencoder = models.Model(inputs, [reconstructed, supervised_out], name="SupervisedAutoencoder")
    encoder = models.Model(inputs, latent, name="Encoder")

    # ----- Class weights for imbalanced data -----
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))

    # Weighted sparse categorical crossentropy
    classes_tf = tf.constant(classes, dtype=tf.int32)
    def weighted_scc(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        indices = tf.searchsorted(classes_tf, y_true)
        weights = tf.gather(tf.constant([class_weights_dict[c] for c in classes], dtype=tf.float32), indices)
        scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return scce * weights

    # ----- Compile model -----
    opt = Adam(learning_rate=learning_rate) if optimizer == 'adam' else SGD(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=opt,
        loss={"reconstruction": "mse", "classification": weighted_scc},
        loss_weights={"reconstruction": 0.5, "classification": 1.0},
        metrics={"classification": "accuracy"}
    )

    return autoencoder, encoder