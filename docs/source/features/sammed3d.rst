====================================================
SAM-Med3D Features
====================================================

Description
============================================

SAM-Med3D features are derived from Segment Anything Model - Medical 3D
(SAM-Med3D), a foundation model for 3D medical image segmentation.
Doi: https://doi.org/10.1109/tnnls.2025.3586694

Purpose
==================

SAM-Med3D features provide:

* Deep learning-based segmentation masks
* Pre-trained feature representations
* Segmentation confidence metrics
* Alternative segmentation approaches

Currently we extract 384 CLS token features from the SAM-Med3D model.
These are black-box features learned by the model during training on large-scale

Current Status
===============================================

This feature category stores segmentation outputs from SAM-Med3D:

* 3D segmentation masks
* Probabilistic segmentation maps
* Segmentation confidence scores

Future Development
===================================================

Planned extensions include:

* Fine-tuned SAM-Med3D models for organoid data
* Feature extraction from learned representations
* Comparison with traditional segmentation methods
* Integration with other foundation models

