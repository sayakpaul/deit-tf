import argparse

import tensorflow as tf
import timm
import torch

torch.set_grad_enabled(False)

import os

from utils import helpers
from vit.deit_models import ViTDistilled
from vit.layers import mha
from vit.model_configs import base_config
from vit.vit_models import ViTClassifier

TF_MODEL_ROOT = "gs://deit-tf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained DeiT weights to TensorFlow."
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="deit_tiny_patch16_224",
        type=str,
        choices=[
            "deit_base_distilled_patch16_224",
            "deit_base_distilled_patch16_384",
            "deit_base_patch16_224",
            "deit_base_patch16_384",
            "deit_small_distilled_patch16_224",
            "deit_small_patch16_224",
            "deit_tiny_distilled_patch16_224",
            "deit_tiny_patch16_224",
        ],
        help="Name of the DeiT model variant.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=224,
        type=int,
        choices=[224, 384],
        help="Image resolution.",
    )
    parser.add_argument(
        "-p",
        "--patch-size",
        default=16,
        type=int,
        help="Patch size.",
    )
    parser.add_argument(
        "-pd",
        "--projection-dim",
        default=192,
        type=int,
        help="Patch projection dimension.",
    )
    parser.add_argument(
        "-nl",
        "--num-layers",
        default=12,
        type=int,
        help="Number of layers denoting depth.",
    )
    parser.add_argument(
        "-nh",
        "--num-heads",
        default=3,
        type=int,
        help="Number of attention heads.",
    )
    return parser.parse_args()


def main(args):
    print(f"Converting {args.model_name}...")
    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(
        model_name=args.model_name, num_classes=1000, pretrained=True
    )
    if "distilled" in args.model_name:
        assert (
            "dist_token" in pt_model.state_dict()
        ), "Distillation token must be present for models trained with distillation."
    pt_model.eval()

    print("Instantiating TF model...")
    model_cls = (
        ViTDistilled if "distilled" in args.model_name else ViTClassifier
    )
    tf_model_config = base_config.get_config(**vars(args))
    tf_model = model_cls(tf_model_config)

    dummy_inputs = tf.ones((2, args.resolution, args.resolution, 3))
    _ = tf_model(dummy_inputs)[0]

    assert tf_model.count_params() == sum(
        p.numel() for p in pt_model.parameters()
    )

    # Load the PT params.
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: pt_model_dict[k].numpy() for k in pt_model_dict}

    print("Beginning parameter porting process...")

    # Projection layers.
    tf_model.layers[0].layers[0] = helpers.modify_tf_block(
        tf_model.layers[0].layers[0],
        pt_model_dict["patch_embed.proj.weight"],
        pt_model_dict["patch_embed.proj.bias"],
    )

    # Positional embedding.
    tf_model.positional_embedding.assign(
        tf.Variable(pt_model_dict["pos_embed"])
    )

    # CLS and (optional) Distillation tokens.
    # Distillation token won't be present in the models trained without distillation.
    tf_model.cls_token.assign(tf.Variable(pt_model_dict["cls_token"]))
    if "distilled" in args.model_name:
        tf_model.dist_token.assign(tf.Variable(pt_model_dict["dist_token"]))

    # Layer norm layers.
    ln_idx = -3 if "distilled" in args.model_name else -2
    tf_model.layers[ln_idx] = helpers.modify_tf_block(
        tf_model.layers[ln_idx],
        pt_model_dict["norm.weight"],
        pt_model_dict["norm.bias"],
    )

    # Head layers.
    head_layer = tf_model.get_layer("classification_head")
    head_layer_idx = -2 if "distilled" in args.model_name else -1
    tf_model.layers[-head_layer_idx] = helpers.modify_tf_block(
        head_layer,
        pt_model_dict["head.weight"],
        pt_model_dict["head.bias"],
    )
    if "distilled" in args.model_name:
        head_dist_layer = tf_model.get_layer("distillation_head")
        tf_model.layers[-1] = helpers.modify_tf_block(
            head_dist_layer,
            pt_model_dict["head_dist.weight"],
            pt_model_dict["head_dist.bias"],
        )

    # Transformer blocks.
    idx = 0

    for outer_layer in tf_model.layers:
        if (
            isinstance(outer_layer, tf.keras.Model)
            and outer_layer.name != "projection"
        ):
            tf_block = tf_model.get_layer(outer_layer.name)
            pt_block_name = f"blocks.{idx}"

            # LayerNorm layers.
            layer_norm_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.LayerNormalization):
                    layer_norm_pt_prefix = (
                        f"{pt_block_name}.norm{layer_norm_idx}"
                    )
                    layer.gamma.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.weight"]
                        )
                    )
                    layer.beta.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.bias"]
                        )
                    )
                    layer_norm_idx += 1

            # FFN layers.
            ffn_layer_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer_pt_prefix = (
                        f"{pt_block_name}.mlp.fc{ffn_layer_idx}"
                    )
                    layer = helpers.modify_tf_block(
                        layer,
                        pt_model_dict[f"{dense_layer_pt_prefix}.weight"],
                        pt_model_dict[f"{dense_layer_pt_prefix}.bias"],
                    )
                    ffn_layer_idx += 1

            # Attention layer.
            for layer in tf_block.layers:
                (q_w, k_w, v_w), (q_b, k_b, v_b) = helpers.get_tf_qkv(
                    f"{pt_block_name}.attn",
                    pt_model_dict,
                    tf_model_config,
                )

                if isinstance(layer, mha.TFViTAttention):
                    # Key
                    layer.self_attention.key = helpers.modify_tf_block(
                        layer.self_attention.key,
                        k_w,
                        k_b,
                        is_attn=True,
                    )
                    # Query
                    layer.self_attention.query = helpers.modify_tf_block(
                        layer.self_attention.query,
                        q_w,
                        q_b,
                        is_attn=True,
                    )
                    # Value
                    layer.self_attention.value = helpers.modify_tf_block(
                        layer.self_attention.value,
                        v_w,
                        v_b,
                        is_attn=True,
                    )
                    # Final dense projection
                    layer.dense_output.dense = helpers.modify_tf_block(
                        layer.dense_output.dense,
                        pt_model_dict[f"{pt_block_name}.attn.proj.weight"],
                        pt_model_dict[f"{pt_block_name}.attn.proj.bias"],
                    )

            idx += 1

    print("Weight population successful, serializing TensorFlow model...")

    save_path = os.path.join(TF_MODEL_ROOT, args.model_name)
    tf_model.save(save_path)
    print(f"TensorFlow model serialized to: {save_path}...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
