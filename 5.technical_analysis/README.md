# Technical analyses

We perform a series of technical analyses to help inform future image acquisition parameters for the 3D organoid profiling pipeline.
These analyses include comparing trade-offs between the number and step-size of z-slices.

## Z-step size

We test in a few well-fovs for a few patients how the z-step size affects the segmentation and the featurization.
We do this by computationally downsampling the z-slices of the original images to simulate larger z-step sizes.
We then segment the downsampled images and compare the segmentation results to the original segmentation results.
We also compute features on the downsampled images (using the full z-step segmentation mask to ensure apples to apples comparison of step size on featurization) and compare them to the features computed on the original images.

Directories of raw/segmented data are named `{patient}_T1-Z-{step}`, where `T1` is timepoint 1 (every
condition tested here is a single timepoint) and `{step}` is the z-step size in micrometers between
consecutive slices. For the pairwise segmentation comparison (below), every condition is downsampled
to match the number of z-slices of the coarsest step tested (1 um), so a comparison labeled
`1um vs 0.5um` means "the 1 um-step mask vs. the 0.5 um-step mask, after both were downsampled to the
same number of slices."

The `compare_z_slice_number_segmentations.ipynb` notebook is slow to re-run (it reads every mask for
every patient/well-fov/compartment), so its computed IOU and volume tables are cached as parquet files
under `processed_data/` (`zslice_iou_results.parquet`, `zslice_per_compartment_volume.parquet`, and
`zslice_per_compartment_volume_normalized.parquet` for the min-max-normalized volumes), and its figures
are saved under `figures/zslice_segmentation_comparison/`.
Downstream visualization/manuscript-figure work should read those cached outputs rather than re-running
the notebook; set `RECOMPUTE = True` inside the notebook only when the underlying masks have changed.

`visualize_zslice_segmentation_comparison.ipynb` (R/ggplot2) reads those same cached parquet files and
re-produces the IOU, volume-ratio, and normalized-volume figures without touching the raw masks or
Python at all — use it to iterate on figure styling/publication-ready plots.

### Table of z-step sizes tested

| patient         | well-fov | z-step size (um) | anisotropy factor | raw                           | segmentation 1                     | segmentation 2                                         |
| --------------- | -------- | ---------------- | ----------------- | ----------------------------- | ---------------------------------- | ------------------------------------------------------ |
| NF0037_T1-Z-0.1 | F4-2     | 0.1              | 1                 | NF0037_T1-Z-0.1/zstack_images | NF0037_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0037_T1-Z-0.2 | F4-2     | 0.2              | 2                 | NF0037_T1-Z-0.2/zstack_images | NF0037_T1-Z-0.2/segmentation_masks | NF0037_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0037_T1-Z-0.5 | F4-2     | 0.5              | 5                 | NF0037_T1-Z-0.5/zstack_images | NF0037_T1-Z-0.5/segmentation_masks | NF0037_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0037_T1-Z-1   | F4-2     | 1.0              | 10                | NF0037_T1-Z-1/zstack_images   | NF0037_T1-Z-1/segmentation_masks   | NF0037_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0037_T1-Z-0.1 | F4-3     | 0.1              | 1                 | NF0037_T1-Z-0.1/zstack_images | NF0037_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0037_T1-Z-0.2 | F4-3     | 0.2              | 2                 | NF0037_T1-Z-0.2/zstack_images | NF0037_T1-Z-0.2/segmentation_masks | NF0037_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0037_T1-Z-0.5 | F4-3     | 0.5              | 5                 | NF0037_T1-Z-0.5/zstack_images | NF0037_T1-Z-0.5/segmentation_masks | NF0037_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0037_T1-Z-1   | F4-3     | 1.0              | 10                | NF0037_T1-Z-1/zstack_images   | NF0037_T1-Z-1/segmentation_masks   | NF0037_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | E4-1-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | E4-1-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | E4-1-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | E4-1-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | E4-5-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | E4-5-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | E4-5-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | E4-5-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | G5-2-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | G5-2-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | G5-2-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | G5-2-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | E4-3-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | E4-3-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | E4-3-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | E4-3-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | G5-1-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | G5-1-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | G5-1-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | G5-1-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | G5-5-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | G5-5-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | G5-5-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | G5-5-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |
| NF0055_T1-Z-0.1 | G5-5-0   | 0.1              | 1                 | NF0055_T1-Z-0.1/zstack_images | NF0055_T1-Z-0.1/segmentation_masks | NA                                                     |
| NF0055_T1-Z-0.2 | G5-5-0   | 0.2              | 2                 | NF0055_T1-Z-0.2/zstack_images | NF0055_T1-Z-0.2/segmentation_masks | NF0055_T1-Z-0.2/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-0.5 | G5-5-0   | 0.5              | 5                 | NF0055_T1-Z-0.5/zstack_images | NF0055_T1-Z-0.5/segmentation_masks | NF0055_T1-Z-0.5/segmentation_masks_from_0_1um_adjusted |
| NF0055_T1-Z-1   | G5-5-0   | 1.0              | 10                | NF0055_T1-Z-1/zstack_images   | NF0055_T1-Z-1/segmentation_masks   | NF0055_T1-Z-1/segmentation_masks_from_0_1um_adjusted   |

## Results

Results are computed per patient, well-fov, and compartment (`organoid`, `nuclei`, `cytoplasm`,
`cell`), comparing each coarser z-step (1 um, 0.5 um, 0.2 um) against the finest step tested
(0.1 um), which is treated as the reference/gold-standard resolution. Three figures are produced
by both `compare_z_slice_number_segmentations.ipynb` (Python) and
`visualize_zslice_segmentation_comparison.ipynb` (R/ggplot2), saved to
`figures/zslice_segmentation_comparison/`:

- **`iou_by_zstep_comparison.png`** — IOU between each coarser step's segmentation mask and the
  0.1 um mask, faceted by compartment and patient, with pairwise Wilcoxon significance (`ns`/`*`/
  `**`/`***`) between the three step-vs-reference comparisons.
- **`volume_ratio_by_zstep_comparison.png`** — the same comparisons, using the ratio of segmented
  volume between each coarser step and the 0.1 um reference.
- **`normalized_volume_by_zstep.png`** — raw segmented volume per z-step (min-max normalized
  within each patient/compartment), again compared against the 0.1 um reference.

Overall, the step-size difference has a fairly small effect on the segmentation results by these metrics: IOU against the 0.1 um reference stays high across most compartments and step sizes, but both IOU and volume show increasing sensitivity as the step size gets coarser (i.e. 1 um differs more from the 0.1 um reference than 0.2 um does), and that sensitivity is not uniform across compartments.
Volume shifts are observed across step sizes, indicating that the segmented volume is sensitive to z-stack resolution to some degree, and the IOU scores confirm that segmentation accuracy itself (not just the resulting volume) can be affected by the z-step size used at acquisition.
See the figures above, or the cached tables in `processed_data/`, for the per-compartment/per-patient breakdown and which comparisons are statistically significant.
