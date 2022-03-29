"""Generates model documentation for DeiT-TF models.

Credits: Willi Gierke
"""

import os
from string import Template

import attr

template = Template(
    """# Module $HANDLE

Fine-tunable DeiT model pre-trained on the $DATASET_DESCRIPTION.

<!-- asset-path: https://storage.googleapis.com/deit-tf/tars/$ARCHIVE_NAME.tar.gz  -->
<!-- task: image-classification -->
<!-- network-architecture: deit -->
<!-- format: saved_model_2 -->
<!-- fine-tunable: true -->
<!-- license: mit -->
<!-- colab: https://colab.research.google.com/github/sayakpaul/deit-tf/blob/main/notebooks/finetune.ipynb -->

## Overview

This model is a DeiT [1] model pre-trained on the $DATASET_DESCRIPTION. You can find the complete
collection of DeiT models on TF-Hub on [this page](https://tfhub.dev/sayakpaul/collections/deit/1).

You can use this model for featue extraction and fine-tuning. Please refer to
the Colab Notebook linked on this page for more details.

## Notes

* The original model weights are provided from [2]. There were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting
steps are available in [3].
* The model can be unrolled into a standard Keras model and you can inspect its topology.
To do so, first download the model from TF-Hub and then load it using `tf.keras.models.load_model`
providing the path to the downloaded model folder.

## References

[1] [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
[2] [DeiT GitHub](https://github.com/facebookresearch/deit)
[3] [DeiT-TF GitHub](https://github.com/sayakpaul/deit-tf)

## Acknowledgements

* [Aritra Roy Gosthipaty](https://github.com/ariG23498)
* [ML-GDE program](https://developers.google.com/programs/experts/)

"""
)


@attr.s
class Config:
    size = attr.ib(type=str)
    dataset = attr.ib(type=str)
    single_resolution = attr.ib(type=int)

    def two_d_resolution(self):
        return f"{self.single_resolution}x{self.single_resolution}"

    def gcs_folder_name(self):
        return f"deit_{self.size}_patch16_{self.single_resolution}_fe"

    def handle(self):
        return (
            f"sayakpaul/deit_{self.size}_patch16_{self.single_resolution}_fe/1"
        )

    def rel_doc_file_path(self):
        """Relative to the tfhub.dev directory."""
        return f"assets/docs/{self.handle()}.md"


for c in [
    Config("tiny", "1k", 224),
    Config("tiny_distilled", "1k", 224),
    Config("small", "1k", 224),
    Config("small_distilled", "1k", 224),
    Config("base", "1k", 224),
    Config("base_distilled", "1k", 224),
    Config("base", "1k", 384),
    Config("base_distilled", "1k", 384),
]:
    dataset_text = "ImageNet-1k dataset"

    save_path = os.path.join(
        "/Users/sayakpaul/Downloads/", "tfhub.dev", c.rel_doc_file_path()
    )
    model_folder = save_path.split("/")[-2]
    model_abs_path = "/".join(save_path.split("/")[:-1])

    if not os.path.exists(model_abs_path):
        os.makedirs(model_abs_path, exist_ok=True)

    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                HANDLE=c.handle(),
                DATASET_DESCRIPTION=dataset_text,
                INPUT_RESOLUTION=c.two_d_resolution(),
                ARCHIVE_NAME=c.gcs_folder_name(),
            )
        )
