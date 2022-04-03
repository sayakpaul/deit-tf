"""
Reference:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/deit.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .vit_models import ViTClassifier


class ViTDistilled(ViTClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens = 2

        # CLS and distillation tokens, positional embedding.
        init_scheme = keras.initializers.TruncatedNormal(
            stddev=self.config.initializer_range
        )
        init_value = init_scheme((1, 1, self.config.projection_dim))
        self.dist_token = tf.Variable(init_value, name="dist_token")
        self.positional_embedding = tf.Variable(
            init_scheme(
                (
                    1,
                    self.config.num_patches + self.num_tokens,
                    self.config.projection_dim,
                )
            ),
            name="pos_embedding",
        )

        # Head layers.
        if not self.config.pre_logits:
            self.head = (
                layers.Dense(
                    self.config.num_classes,
                    kernel_initializer="zeros",
                    name="classification_head",
                )
                if self.config.num_classes > 0
                else tf.nn.identity
            )
            self.head_dist = (
                layers.Dense(
                    self.config.num_classes,
                    kernel_initializer="zeros",
                    name="distillation_head",
                )
                if self.config.num_classes > 0
                else tf.nn.identity
            )

    def call(self, inputs, training=True):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append the tokens.
        cls_token = tf.tile(self.cls_token, (n, 1, 1))
        dist_token = tf.tile(self.dist_token, (n, 1, 1))
        if cls_token.dtype != projected_patches.dtype != dist_token.dtype:
            cls_token = tf.cast(cls_token, projected_patches.dtype)
            dist_token = tf.cast(dist_token, projected_patches.dtype)
        projected_patches = tf.concat(
            [cls_token, dist_token, projected_patches], axis=1
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
        if self.config.pre_logits:
            return (
                representation[:, 0] + representation[:, 1]
            ) / 2, attention_scores

        # Classification heads.
        else:
            x, x_dist = self.head(representation[:, 0]), self.head_dist(
                representation[:, 1]
            )

            if "distilled" in self.config.name and training:
                # Only return separate classification predictions when training in distilled mode.
                return x, x_dist, attention_scores

            else:
                # During standard train / finetune, inference average the classifier predictions.
                # Additionally, return the attention scores too.
                return (x + x_dist) / 2, attention_scores
