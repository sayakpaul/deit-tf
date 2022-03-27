# DeiT-TF (Data-efficient Image Transformers)

This repository provides TensorFlow / Keras implementations of different DeiT
[1] variants from Touvron et al. It also provides the TensorFlow / Keras models that have been
populated with the original DeiT pre-trained params available from [2]. These
models are not blackbox SavedModels i.e., they can be fully expanded into `tf.keras.Model`
objects and one can call all the utility functions on them (example: `.summary()`).

As of today, all the TensorFlow / Keras variants of the **DeiT** models listed
[here](https://github.com/facebookresearch/deit#model-zoo) are available in this
repository.

Refer to the ["Using the models"](https://github.com/sayakpaul/deit-tf#using-the-models)
section to get started. 

## Conversion

TensorFlow / Keras implementations are available in `vit/vit_models.py` and `vit/deit_models.py`.
Conversion utilities are in `convert.py`.

## Models

The converted models will be available on on TF-Hub soon. 

TODO: Code for summarizing for model.

## Results

Results are on ImageNet-1k validation set (top-1 accuracy). 

TODO

## Using the models

**Pre-trained models**:

* Off-the-shelf classification: [Colab Notebook](TODO)
* Fine-tuning: [Colab Notebook](TODO)
 
 **Randomly initialized models**:
 
 ```py
from vit.model_configs import base_config
from vit.deit_models import ViTDistilled
 
 distilled_tiny_tf_config = base_config.get_config(
    name="deit_tiny_distilled_patch16_224"
)
deit_tiny_distilled_patch16_224 = ViTDistilled(distilled_tiny_tf_config)

dummy_inputs = tf.ones((2, 224, 224, 3))
_ = deit_tiny_distilled_patch16_224(dummy_inputs)
print(deit_tiny_distilled_patch16_224.summary(expand_nested=True))
 ```
 
 To view different model configurations, refer to `convert_all_models.py`.
 
## Upcoming (contributions welcome)

- [ ] Align layer initializers (useful if someone wanted to train the models
from scratch) 
- [ ] Fine-tuning notebook 
- [ ] Off-the-shelf-classification notebook
- [ ] Publish models on TF-Hub

## References

[1] DeiT paper: https://arxiv.org/abs/2012.12877

[2] Official DeiT code: https://github.com/facebookresearch/deit

## Acknowledgements

* [`timm` library source code](https://github.com/rwightman/pytorch-image-models)
* [ML-GDE program](https://developers.google.com/programs/experts/)
