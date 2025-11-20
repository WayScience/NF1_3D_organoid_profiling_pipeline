# NF1 3D Organoid Profiling Pipeline
Patients living with Neurofibromatosis Type 1 (NF1) often develop neurofibromas (NFs), which are complex benign tumors.
However, there are only two FDA-approved therapies for NF1-associated inoperable plexiform neurofibromas (PNFs): Mirdametinib and Selumetinib.
Thus, we **urgently need more therapeutic options** for neurofibromas.

To address this, we have developed a 3D patient-derived tumor organoid model of NF1.
We developed a modified 3D Cell Painting protocol to generate high-content imaging data from these organoids.
This repository contains the code and documentation for a comprehensive analysis pipeline to process and analyze these 3D organoid models of NF1 NFs.

---
### Raw channels
| 405 | 488 | 555 | 640 |
|:-:|:-:|:-:|:-:|
| <img src="./2.segment_images/animations/media_for_readme/C4-2_C4-2_405_animation.gif" alt="GIF 1"> | <img src="./2.segment_images/animations/media_for_readme/C4-2_C4-2_488_animation.gif"  alt="GIF 2"> | <img src="./2.segment_images/animations/media_for_readme/C4-2_C4-2_555_animation.gif" alt="GIF 3"> | <img src="./2.segment_images/animations/media_for_readme/C4-2_C4-2_640_animation.gif" alt="GIF 4"> |

### Organoid, Nuclei, Cell, and Cytoplasm Segmentations
| Organoid | Nuclei | Cell | Cytoplasm |
|:-:|:-:|:-:|:-:|
| <img src="./2.segment_images/animations/media_for_readme/C4-2_organoid_masks (labels)_animation.gif" alt="GIF 1"> | <img src="./2.segment_images/animations/media_for_readme/C4-2_nuclei_masks (labels)_animation.gif" alt="GIF 2"> | <img src="./2.segment_images/animations/media_for_readme/C4-2_cell_masks (labels)_animation.gif" alt="GIF 3"> | <img src="./2.segment_images/animations/media_for_readme/C4-2_cytoplasm_mask (labels)_animation.gif" alt="GIF 4">
---

## Cell Painting
The modified 3D cell painting assay was performed on patient-derived NF1 tumor organoids using the following stains:
| Stain | Target | Channel |
|-------|--------|---------------------|
| Hoechst | Nuclei | 405 nm |
| Concanavalin A | Endoplasmic Reticulum | 488 nm |
| Phalloidin | F-actin | 555 nm |
| Wheat Germ Agglutinin | Golgi Apparatus & Plasma Membrane | 555 nm |
| MitoTracker | Mitochondria | 640 nm |
| - | - | Brightfield |

Cell Painting was performed in 96-well plates with multiple fields of view (FOVs) imaged per well across multiple z-planes.

## Imaging
We imaged the stained organoids using an spinning disk confocal microscope with pixel size of 0.100 µm and z-step size of 1 µm.

## Repository structure
Top-level directories and a short description of what they contain:

- 0.preprocessing_data
  - Scripts and notebooks for raw data ingest, file-structure setup, z‑stack creation and optional deconvolution.
- 1.image_quality_control
  - CellProfiler/analysis notebooks and scripts for QC and imaging QA.
- 2.segment_images
  - Full segmentation pipeline: notebooks, scripts for 2.5D/3D segmentation, mask reconstruction and post‑hoc refinements.
- 3.cellprofiling
  - CellProfiler pipelines, wrapper scripts and helpers to extract per-cell/per-organoid features.
- 4.processing_image_based_profiles
  - Scripts to merge, normalize, filter and aggregate image‑based profiles across wells/patients.
- 5.EDA
  - Exploratory analyses, visualization notebooks and scripts (UMAP, mAP, hit detection, figures).
- 6.dynamic_viz_Rshiny
  - Shiny app and deployment helpers for interactive result exploration.
- 7.technical_analysis
  - Experimental/analysis scripts and ad‑hoc notebooks used for method development.
- data
  - Expected location for processed image files, intermediate artifacts and aggregated profiles (patient subfolders).
- src / utils
  - Utility libraries and helper modules used across notebooks and scripts.
- environments
  - Conda/Mamba environment YAMLs and Makefile targets to reproduce required environments.

**Notes:**
- Each module/directory generally includes a README with more detailed run instructions — consult those before running.
- Many batch/HPC runs use notebook-to-script conversion and wrapper shell scripts; see the per-module README for exact commands and expected inputs.

## Using this repository
### This is an open-source repository that is a work in progress.
### We do have future plans to pipeline this work into a more user-friendly and scalable format.
This work as it stands now is optimized to be run our Institution's HPC (SLURM) environment and may require some adaptation to run in other environments.
Further, some code and pathing is specific to our lab's Network Attached Storage (NAS) structure.
We have written in code to decouple our structure and will continue to work on scaling, portability, interoperability, and usability.
Please feel free to reach out with any questions or issues via GitHub Issues or contact the project lead: [Michael J. Lippincott](https://mikelippincott.github.io/) or the corresponding author: [Gregory P. Way](https://github.com/gwaybio).

### Environment Setup
We recommend using `mamba` or `conda` to create the required environments.
We have written a `makefile` to help with this process.
```bash
cd environments || exit
make --always-make
cd .. || exit
```

### System Requirements
- Linux-based OS
- HPC/SLURM environment recommended for large-scale runs
- At least multiple TBs of storage for raw and processed images
- Sufficient RAM and CPU/GPU resources depending on dataset size
    - We recommend at least 128GB RAM and multiple CPU cores for image processing steps
    - Though we have been able to get RAM usage under 8GB per well_fov by distubuting the compute.
        - Please note that this RAM usage is highly dependent on the number of z-slices, image dimensions, and number of channels.
        - Here we generally have 30-50 z-slices, ~1500x1500 pixel images, and 4 channels. We rarly exceed 100 z-slices. Additionally scaling in z-slices will require more compute time and RAM.
- Optional: GPU resources for segmentation and deep learning based feature extraction
    - We have found that a NVIDIA 3090 TI (24GB VRAM) is more than enough for our segmentation tasks.
    - It is important to note that part of the advantage of using 2.5D segmentation is that it greatly reduces the GPU VRAM requirements compared to full 3D segmentation - especially as z-slice count scales up.
### Data Availability
The raw and processed imaging data are not quite publicly available at this time.
We will have data available at some point on the NF Data Portal via synapse.

