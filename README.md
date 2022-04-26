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
section to get started. You can also follow along with this tutorial: https://keras.io/examples/vision/deit/.

**Updates**

* April 22, 2022: This project won the [#TFCommunitySpotlight award](https://twitter.com/TensorFlow/status/1516869315517198337).

## Table of contents

* [Conversion](https://github.com/sayakpaul/deit-tf#conversion)
* [Collection of pre-trained models (converted from PyTorch to TensorFlow)](https://github.com/sayakpaul/deit-tf#models)
* [Results of the converted models](https://github.com/sayakpaul/deit-tf#results)
* [How to use the models?](https://github.com/sayakpaul/deit-tf#using-the-models)
* [Training with DeiT](https://github.com/sayakpaul/deit-tf#training-with-deit)
* [References](https://github.com/sayakpaul/deit-tf#references)
* [Acknowledgements](https://github.com/sayakpaul/deit-tf#acknowledgements)

## Conversion

TensorFlow / Keras implementations are available in `vit/vit_models.py` and `vit/deit_models.py`.
Conversion utilities are in `convert.py`.

## Models

Find the models on TF-Hub here: https://tfhub.dev/sayakpaul/collections/deit/1. You can fully inspect the
architecture of the TF-Hub models like so:

```py
import tensorflow as tf

model_gcs_path = "gs://tfhub-modules/sayakpaul/deit_tiny_patch16_224/1/uncompressed"
model = tf.keras.models.load_model(model_gcs_path)

dummy_inputs = tf.ones((2, 224, 224, 3))
_ = model(dummy_inputs)
print(model.summary(expand_nested=True))
```

## Results

Results are on ImageNet-1k validation set (top-1 accuracy). 

|    | **model_name**                   |   **top1_acc(%)** |   **top5_acc(%)** |   **orig_top1_acc(%)** |   **orig_top5_acc(%)** |
|---:|:---------------------------------|--------------:|--------------:|-------------------:|-------------------:|
|  0 | deit_tiny_patch16_224            |        72.136 |        91.128 |               72.2 |               91.1 |
|  1 | deit_tiny_distilled_patch16_224  |        74.522 |        91.896 |               74.5 |               91.9 |
|  2 | deit_small_patch16_224           |        79.828 |        94.954 |               79.9 |               95   |
|  3 | deit_small_distilled_patch16_224 |        81.172 |        95.414 |               81.2 |               95.4 |
|  4 | deit_base_patch16_224            |        81.798 |        95.592 |               81.8 |               95.6 |
|  5 | deit_base_patch16_384            |        82.894 |        96.234 |               82.9 |               96.2 |
|  6 | deit_base_distilled_patch16_224  |        83.326 |        96.496 |               83.4 |               96.5 |
|  7 | deit_base_distilled_patch16_384  |        85.238 |        97.172 |               85.2 |               97.2 |

Results can be verified with the code in `i1k_eval`. Original results were sourced from [2].


## Using the models

**Pre-trained models**:

* Off-the-shelf classification: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/deit-tf/blob/main/notebooks/classification.ipynb)
* Fine-tuning: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/deit-tf/blob/main/notebooks/finetune.ipynb)

These models also output attention weights from each of the Transformer blocks.
Refer to [this notebook](https://colab.research.google.com/github/sayakpaul/deit-tf/blob/main/notebooks/classification.ipynb)
for more details. Additionally, the notebook shows how to visualize the attention maps for a given image.

<br>

<div align=center>
 <img src="https://i.ibb.co/hHzggDr/attention-map.png" width=700/>
</div>

<br>
 
**Randomly initialized models**:
 
```py
from vit.model_configs import base_config
from vit.deit_models import ViTDistilled

import tensorflow as tf
 
distilled_tiny_tf_config = base_config.get_config(
    name="deit_tiny_distilled_patch16_224"
)
deit_tiny_distilled_patch16_224 = ViTDistilled(distilled_tiny_tf_config)

dummy_inputs = tf.ones((2, 224, 224, 3))
_ = deit_tiny_distilled_patch16_224(dummy_inputs)
print(deit_tiny_distilled_patch16_224.summary(expand_nested=True))
```

To view different model configurations, refer to `convert_all_models.py`.


## Training with DeiT

You can refer to the `notebooks/deit-trainer.ipynb` notebok to get a sense of how distillation
is actually performed using DeiT. Additionally, that notebook also provides code in case you
wanted to train a model from scratch instead of distillation.
 

## References

[1] DeiT paper: https://arxiv.org/abs/2012.12877

[2] Official DeiT code: https://github.com/facebookresearch/deit

## Acknowledgements

* [Aritra Roy Gosthipaty](https://github.com/ariG23498) who was instrumental in implementing
some parts of the core ViT module (`vit.vit_models`) for another project.
* [`timm` library source code](https://github.com/rwightman/pytorch-image-models)
for the awesome codebase.
* [ML-GDE program](https://developers.google.com/programs/experts/) for
providing GCP credits that supported my experiments.
