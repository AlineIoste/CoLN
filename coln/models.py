import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def binary_focal_loss(gamma: float = 1.5, alpha: float = 0.35):
    """Binary focal loss (for imbalanced classification)."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        fl = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)
    return loss

def build_mlp(input_dim: int, lr: float = 3e-4) -> tf.keras.Model:
    """MLP architecture used in the notebook/paper (Keras)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, kernel_regularizer=l2(1e-4)),
        LayerNormalization(),
        Activation("relu"),
        Dropout(0.25),
        Dense(128, kernel_regularizer=l2(1e-4)),
        LayerNormalization(),
        Activation("relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=binary_focal_loss(gamma=1.5, alpha=0.35),
        metrics=["accuracy"],
    )
    return model
