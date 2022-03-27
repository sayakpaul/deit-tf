import os

MODEL_CONFIGS = {
    "deit_base_distilled_patch16_224": {
        "dim": 768,
        "num_layers": 12,
        "num_heads": 3,
    },
    "deit_base_distilled_patch16_384": {
        "dim": 768,
        "num_layers": 12,
        "num_heads": 6,
    },
    "deit_base_patch16_224": {
        "dim": 768,
        "num_layers": 12,
        "num_heads": 12,
    },
    "deit_base_patch16_384": {
        "dim": 768,
        "num_layers": 12,
        "num_heads": 12,
    },
    "deit_small_distilled_patch16_224": {
        "dim": 384,
        "num_layers": 12,
        "num_heads": 6,
    },
    "deit_small_patch16_224": {
        "dim": 384,
        "num_layers": 12,
        "num_heads": 6,
    },
    "deit_tiny_distilled_patch16_224": {
        "dim": 192,
        "num_layers": 12,
        "num_heads": 3,
    },
    "deit_tiny_patch16_224": {
        "dim": 192,
        "num_layers": 12,
        "num_heads": 3,
    },
}


def main():
    for model_name in MODEL_CONFIGS.keys():
        image_sz = int(model_name.split("_")[-1])
        patch_sz = int(model_name.split("_")[-2][-2:])

        model_config = MODEL_CONFIGS.get(model_name)
        proj_dim = model_config.get("dim")
        num_layers = model_config.get("num_layers")
        num_heads = model_config.get("num_heads")

        command = f"python convert.py -m {model_name} -r {image_sz} -p {patch_sz} -pd {proj_dim} -nl {num_layers} -nh {num_heads}"
        os.system(command)


if __name__ == "__main__":
    main()
