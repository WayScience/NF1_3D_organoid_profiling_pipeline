# CHAMMI-75 Features

## Description

CHAMMI-75 features are deep learning-based embeddings extracted using
[MorphEm](https://huggingface.co/CaicedoLab/MorphEm), a self-supervised Vision
Transformer (ViT) model pre-trained on the CHAMMI benchmark dataset of
fluorescence microscopy images.

DOI: <https://doi.org/10.1038/s41592-024-02349-9>

## Architecture

The model uses a **Bag-of-Channels (BoC)** strategy:

* Each fluorescence channel is treated as an independent grayscale image
* The single channel is replicated into 3 copies to satisfy the ViT's RGB input requirement
* The ViT encoder processes each channel independently
* The **CLS token** output (384-dimensional) is extracted as the feature vector per channel per object

```{mermaid}
graph TD
    A["Cropped 2D image Y × X, single channel"] --> B["Replicate channel→ 3, Y, X"]
    B --> C["SaturationNoiseInjector PerImageNormalize Resize 224×224"]
    C --> D["ViT Encoder MorphEm"]
    D --> E["CLS token 384-dim embedding"]
```

## Pre-processing Pipeline

As CHAMMI recommends, before passing images to the model, we apply three transforms in sequence:

1. **SaturationNoiseInjector** – Saturated pixels (value = 255) in the input
   channel are replaced with uniform random noise sampled from `[200, 255]`.
   This prevents the model from learning artefacts caused by pixel saturation.

2. **PerImageNormalize** – Each image shape and format is normalized independently using
   `InstanceNorm2d`.

3. **Resize** – The image is resized to 224 × 224 pixels to match the ViT
   input resolution.

## Features Extracted

| Feature | Description |
|---------|-------------|
| CHAMMI1 – CHAMMI384 | CLS-token embedding dimensions from the MorphEm ViT encoder |

Currently **384 features** are extracted per channel per object.

## Applications

CHAMMI-75 features are useful for:

* Capturing phenotypes that are missed by hand-crafted features.
* Identifying subtle treatment effects in fluorescence images.
* Downstream classification tasks.

## References

* <https://arxiv.org/abs/2512.20833>
* Hugging Face model card: <https://huggingface.co/CaicedoLab/MorphEm>
