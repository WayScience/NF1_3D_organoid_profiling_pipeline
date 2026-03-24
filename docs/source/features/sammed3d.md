# SAM-med3d features

## Description

SAM-med3d features are derived from segment anything model - medical 3D (SAM-med3d), a foundation model for 3D medical image segmentation. doi: https://doi.org/10.1109/tnnls.2025.3586694

## Purpose

SAM-med3d features provide:

- Deep learning-based segmentation masks
- Pre-trained feature representations
- Segmentation confidence metrics
- Alternative segmentation approaches

Currently we extract 384 CLS token features from the SAM-med3d model. these are black-box features learned by the model during training on large-scale medical image datasets.

## Current status

This feature category stores segmentation outputs from SAM-med3d:

- 3D segmentation masks
- Probabilistic segmentation maps
- Segmentation confidence scores

## Future development

Planned extensions include:

- Fine-tuned SAM-med3d models for organoid data
- Feature extraction from learned representations
- Comparison with traditional segmentation methods
- Integration with other foundation models
