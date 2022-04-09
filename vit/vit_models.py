"""
Vision Transformer model class put together by
Aritra (ariG23498) and Sayak (sayakpaul)

Reference:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from typing import List

import tensorflow as tf
from ml_collections import ConfigDict
from tensorflow import keras
from tensorflow.keras import layers

from .layers.ls import LayerScale
from .layers.mha import TFViTAttention
from .layers.sd import StochasticDepth


def mlp(x: int, dropout_rate: float, hidden_units: List[int]):
    """FFN for a Transformer block."""
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for (idx, units) in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=tf.nn.gelu if idx == 0 else None,
            bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer(config: ConfigDict, name: str, drop_prob=0.0) -> keras.Model:
    """Transformer block with pre-norm."""
    num_patches = (
        config.num_patches + 2
        if "distilled" in config.name
        else config.num_patches + 1
    )
    if "distilled" in config.name:
        num_patches = config.num_patches + 2
    elif "distilled" not in config.name and config.classifier == "token":
        num_patches = config.num_patches + 1
    elif (
        config.classifer == "gap"
    ):  # This setting should not be used during weight porting.
        assert (
            "distilled" not in config.name
        ), "Distillation token is not suitable for GAP."
        num_patches = config.num_patches + 0

    encoded_patches = layers.Input((None, config.projection_dim))

    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(
        encoded_patches
    )

    # Multi Head Self Attention layer 1.
    attention_output, attention_score = TFViTAttention(config)(
        x1, output_attentions=True
    )
    attention_output = (
        LayerScale(config)(attention_output)
        if config.init_values
        else attention_output
    )
    attention_output = (
        StochasticDepth(drop_prob)(attention_output)
        if drop_prob
        else attention_output
    )

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)

    # MLP layer 1.
    x4 = mlp(
        x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate
    )
    x4 = LayerScale(config)(x4) if config.init_values else x4
    x4 = StochasticDepth(drop_prob)(x4) if drop_prob else x4

    # Skip connection 2.
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, [outputs, attention_score], name=name)


class ViTClassifier(keras.Model):
    """Vision Transformer base class."""

    def __init__(self, config: ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Patchify + embedding.
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=config.projection_dim,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    name="conv_projection",
                    kernel_initializer="lecun_normal",
                ),
                layers.Reshape(
                    target_shape=(-1, config.projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        # Positional embedding.
        init_scheme = keras.initializers.TruncatedNormal(
            stddev=config.initializer_range
        )
        init_shape = (
            1,
            config.num_patches + 1
            if self.config.classifier == "token"
            else config.num_patches,
            config.projection_dim,
        )

        self.positional_embedding = tf.Variable(
            init_scheme(init_shape), name="pos_embedding"
        )  # This will be loaded with the pre-trained positional embeddings later.

        # Transformer blocks.
        dpr = [
            x
            for x in tf.linspace(
                0.0, self.config.drop_path_rate, self.config.num_layers
            )
        ]
        self.transformer_blocks = [
            transformer(config, name=f"transformer_block_{i}", drop_prob=dpr[i])
            for i in range(config.num_layers)
        ]

        # CLS token or GAP.
        if config.classifier == "token":
            init_scheme = keras.initializers.RandomNormal(stddev=1e-6)
            initial_value = init_scheme((1, 1, config.projection_dim))
            self.cls_token = tf.Variable(
                initial_value=initial_value, trainable=True, name="cls"
            )

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        # Other layers.
        self.dropout = layers.Dropout(config.dropout_rate)
        self.layer_norm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps
        )
        if not self.config.pre_logits:
            self.head = layers.Dense(
                config.num_classes,
                kernel_initializer="zeros",
                dtype="float32",
                name="classification_head",
            )

    def call(self, inputs, training=None):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append class token if needed.
        if self.config.classifier == "token":
            cls_token = tf.tile(self.cls_token, (n, 1, 1))
            if cls_token.dtype != projected_patches.dtype:
                cls_token = tf.cast(cls_token, projected_patches.dtype)
            projected_patches = tf.concat(
                [cls_token, projected_patches], axis=1
            )

        # Add positional embeddings to the projected patches.
        encoded_patches = (
            self.positional_embedding + projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        # Initialize a dictionary to store attention scores from each transformer
        # block.
        attention_scores = dict()

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches, attention_score = transformer_module(
                encoded_patches
            )
            attention_scores[f"{transformer_module.name}_att"] = attention_score

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        if self.config.classifier == "token":
            encoded_patches = representation[:, 0]
        elif self.config.classifier == "gap":
            encoded_patches = self.gap_layer(representation)

        if self.config.pre_logits:
            return encoded_patches, attention_scores

        # Classification head.
        else:
            output = self.head(encoded_patches)
            return output, attention_scores
