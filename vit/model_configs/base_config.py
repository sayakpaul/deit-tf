import ml_collections


def get_config(
    model_name: str = "deit_tiny_patch16_224",
    resolution: int = 224,
    patch_size: int = 16,
    projection_dim: int = 192,
    num_layers: int = 12,
    num_heads: int = 3,
    init_values: float = None,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    pre_logits: bool = False,
) -> ml_collections.ConfigDict:
    """Default initialization refers to deit_tiny_patch16_224 for ImageNet-1k.

    Reference:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/deit.py#L141
    """
    config = ml_collections.ConfigDict()
    config.name = model_name

    config.input_shape = (resolution, resolution, 3)
    config.image_size = resolution
    config.patch_size = patch_size
    config.num_patches = (config.image_size // config.patch_size) ** 2
    config.num_classes = 1000

    config.initializer_range = 0.02
    config.layer_norm_eps = 1e-6
    config.projection_dim = projection_dim
    config.num_heads = num_heads
    config.num_layers = num_layers
    config.mlp_units = [
        config.projection_dim * 4,
        config.projection_dim,
    ]
    config.dropout_rate = dropout_rate
    config.classifier = "token"
    config.init_values = init_values
    config.drop_path_rate = drop_path_rate

    config.pre_logits = pre_logits

    return config.lock()
