import tensorflow as tf
from ml_collections import ConfigDict
from tensorflow.keras import layers


# Referred from: github.com:rwightman/pytorch-image-models.
class LayerScale(layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(
            config.init_values * tf.ones((config.projection_dim,)),
            name="layer_scale",
        )

    def call(self, x):
        return x * self.gamma
