# Technical analysis

Technical analysis of acquisiton parameters like z-slice number and z-step size, and pixel size.
This analysis is intended to help inform future acquisition parameters for the 3D organoid profiling pipeline.

## Z-step size

We test in a few well-fovs for a few patients how the z-step size affects the segmentation and the featurization.
We do this by computationally downsampling the z-slices of the original images to simulate larger z-step sizes.
We then segment the downsampled images and compare the segmentation results to the original segmentation results.
We also compute features on the downsampled images and compare them to the features computed on the original images.

### Table of z-step sizes tested

| patient         | well-fov | z-step size (um) | anisotropy factor | raw                           | segmentation                       |
| --------------- | -------- | ---------------- | ----------------- | ----------------------------- | ---------------------------------- |
| NF0037_T1-Z-0.1 | F4-2     | 0.1              | 1                 | NF0037_T1-Z-0.1/zstack_images | NF0037_T1-Z-0.1/segmentation_masks |
| NF0037_T1-Z-0.2 | F4-2     | 0.2              | 1                 | NF0037_T1-Z-0.2/zstack_images | NF0037_T1-Z-0.2/segmentation_masks |
| NF0037_T1-Z-0.5 | F4-2     | 0.5              | 1                 | NF0037_T1-Z-0.5/zstack_images | NF0037_T1-Z-0.5/segmentation_masks |
| NF0037_T1-Z-1   | F4-2     | 1.0              | 1                 | NF0037_T1-Z-1/zstack_images   | NF0037_T1-Z-1/segmentation_masks   |
| NF0037_T1-Z-0.1 | F4-3     | 0.1              | 1                 | NF0037_T1-Z-0.1/zstack_images | NF0037_T1-Z-0.1/segmentation_masks |
| NF0037_T1-Z-0.2 | F4-3     | 0.2              | 1                 | NF0037_T1-Z-0.2/zstack_images | NF0037_T1-Z-0.2/segmentation_masks |
| NF0037_T1-Z-0.5 | F4-3     | 0.5              | 1                 | NF0037_T1-Z-0.5/zstack_images | NF0037_T1-Z-0.5/segmentation_masks |
| NF0037_T1-Z-1   | F4-3     | 1.0              | 1                 | NF0037_T1-Z-1/zstack_images   | NF0037_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | E4-1-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | E4-1-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | E4-1-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | E4-1-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | E4-5-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | E4-5-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | E4-5-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | E4-5-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | G5-2-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | G5-2-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | G5-2-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | G5-2-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | E4-3-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | E4-3-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | E4-3-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | E4-3-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | G5-1-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | G5-1-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | G5-1-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | G5-1-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | G5-5-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | G5-5-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | G5-5-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | G5-5-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
| NF0055_T1-Z-0.1 | G5-5-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks |
| NF0055_T1-Z-0.2 | G5-5-0   | 0.2              | 1                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks |
| NF0055_T1-Z-0.5 | G5-5-0   | 0.5              | 1                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks |
| NF0055_T1-Z-1   | G5-5-0   | 1.0              | 1                 | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   |
