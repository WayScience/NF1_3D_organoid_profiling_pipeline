===========================================================================================
NF1 3D Organoid Profiling Pipeline: Complete Workflow
===========================================================================================

Overview
==================================================

The NF1 3D Organoid Profiling Pipeline is a comprehensive end-to-end system for processing raw 3D microscopy imaging data through segmentation, feature extraction, quality control, and statistical analysis. This document provides a complete guide to the entire workflow.

.. image:: _static/workflow_diagram.png
   :alt: NF1 Pipeline Workflow
   :align: center
   :width: 100%

Pipeline Architecture
==================================================

The pipeline follows a hierarchical processing structure:

**Execution Strategy:**

- SLURM-based HPC scheduling for parallel processing
- Conditional execution based on file existence
- Automatic job submission throttling (max 990 concurrent jobs)

Detailed Workflow Stages
==================================================

Stage 0: Data Preprocessing
--------------------------------------------------

**Directory:** ``0.preprocessing_data/``

**Purpose:** Transform raw microscopy data into standardized 3D z-stack images ready for analysis.

Process Steps
^^^^^^^^^^^^^^

1. **Patient-Specific Preprocessing**

   - Organize raw image files by patient ID
   - Validate file naming conventions
   - Create patient-specific directory structures

2. **File Structure Updates**

   - Standardize directory hierarchy across patients
   - Rename files to consistent naming scheme
   - Verify data integrity

3. **Z-Stack Creation**

   - Combine 2D image slices into 3D z-stacks
   - Handle different microscope formats (CQ1, IX83)
   - Maintain metadata and channel information
   - Typical z-spacing: 0.5 µm
   - Stack depth: 50-100 slices (~25-50 µm)

4. **Corruption Detection**

   - Validate TIFF file integrity
   - Check for incomplete or damaged files
   - Flag problematic datasets

5. **Deconvolution Preprocessing**

   - Prepare images for Huygens deconvolution
   - Generate parameter files
   - Organize batch processing structure

6. **Post-Deconvolution Processing**

   - Import deconvolved images
   - Verify output quality
   - Update file paths and metadata

**Inputs:**

- Raw 2D TIFF images from microscope (5 channels × N z-slices × M wells)
- Metadata files (experiment design, plate layouts)

**Outputs:**

- Deconvolved 3D z-stack TIFF files organized by patient/well/FOV
- File structure: ``data/{patient}/zstack_images/{well_fov}/{channel}.tif``

**Key Parameters:**

- Objective: 60x/1.35 NA oil immersion
- Oil RI: 1.518
- Voxel size: ~0.1 µm (XY) × 1 µm (Z)

**Execution:**

.. code-block:: bash

   cd 0.preprocessing_data
   # For CQ1 microscope data
   python scripts/1z.make_zstack_and_copy_over_CQ1.py --patient NF0014_T1

   python scripts/1.make_zstack_and_copy_over.py --patient NF0014_T1

Stage 1: Image Quality Control
--------------------------------------------------

**Directory:** ``1.image_quality_control/``

**Purpose:** Assess image quality and flag problematic well FOVs before segmentation.

Process Steps
^^^^^^^^^^^^^^

1. **CellProfiler QC Pipeline**

   - Extract whole-image metrics using CellProfiler
   - Compute per-slice statistics
   - Export metrics to CSV

2. **Blur Evaluation**

   - Calculate Laplacian variance for focus detection
   - Identify out-of-focus z-slices
   - Set thresholds for acceptable sharpness

3. **Saturation Analysis**

   - Detect overexposed pixels per channel
   - Calculate percentage of saturated voxels
   - Flag wells with excessive saturation (>5%)

4. **QC Report Generation**

   - Create visualizations with ggplot2 (R)
   - Generate per-plate and per-patient summaries
   - Produce pass/fail flags for each well FOV

**Quality Metrics:**

- **Blur:** Laplacian variance, focus score
- **Saturation:** Percentage of clipped pixels
- **Signal-to-Noise:** Mean signal / background std
- **Illumination:** Uniformity across FOV

**Inputs:**

- Deconvolved z-stack images from Stage 0

**Outputs:**

- QC flags file: ``data/{patient}/qc_flags.csv``
- QC reports: HTML/PDF summaries with plots
- Flagged well list for exclusion from downstream analysis

**Execution:**

.. code-block:: bash

   cd 1.image_quality_control
   jupyter nbconvert --to notebook --execute notebooks/*.ipynb

Stage 2: Image Segmentation
--------------------------------------------------

**Directory:** ``2.segment_images/``

**Purpose:** Generate 3D masks for nuclei, cells, organoids, and cytoplasm compartments.

Process Steps
^^^^^^^^^^^^^^

1. **Nuclei Segmentation**

   - Use Cellpose 4.0
   - Process DNA channel (405 nm)

2. **Organoid Segmentation**

   - Use cellpose 3.x using a custom size invariant search algorithm

3. **Cell Segmentation**

   - Segment individual cells using F-actin and AGP channel (555 nm)
   - Expand from nuclear seeds
   - Use 3D watershed for cell boundary detection

4. **Cytoplasm Derivation**

   - Subtract nuclear masks from cell masks
   - Generate cytoplasmic compartment masks

5. **Mask Refinement**

   - stitch 2D masks into 3D volumes
   - match objects to retain the same IDs across z-slices
   - pair nuclei to cells and organoids

**Execution:**

.. code-block:: bash

   cd 2.segment_images
   sbatch grand_parent_segmentation.sh

Stage 3: Feature Extraction
--------------------------------------------------

**Directory:** ``3.cellprofiling/``

**Purpose:** Extract morphological, intensity, and texture features from segmented objects.

Process Steps
^^^^^^^^^^^^^^

The featurization follows a hierarchical job submission structure:

1. **Grandparent Process** (``run_featurization_grandparent.sh``)

   - Loops through all well FOVs for a patient
   - Submits parent jobs per well FOV

2. **Parent Process** (``run_featurization_parent.sh``)

   - Loops through feature types × compartments × channels
   - Submits child jobs for each combination

3. **Child Process** (Individual feature extraction scripts)

   - Executes specific feature calculation
   - Saves output as parquet

**Feature Types:**

For more details on feature types and extraction methods, refer to
``extraction_math`` or the ``features/`` documentation.

Stage 4: Profile Processing
--------------------------------------------------

**Directory:** ``4.processing_image_based_profiles/``

**Purpose:** Merge, normalize, and aggregate features across wells and patients.

Process Steps:
^^^^^^^^^^^^^^

1. **Feature Merging**

   - Combine all feature CSVs per well FOV
   - Use cytotable for SQLite → Parquet conversion
   - Create single-cell (sc) and organoid-level profiles

2. **Annotation**

   - Add treatment metadata from plate maps
   - Link drug names, targets, concentrations
   - Add patient genotype information

3. **Normalization**

   - Z-score normalization per plate
   - Standardize features: ``(x - μ) / σ``
   - Handle batch effects

4. **Feature Selection**

   - Remove low-variance features
   - Filter correlated features (correlation > 0.9)
   - Drop blocklisted features
   - Apply frequency cutoff for categorical features

5. **Aggregation**

   - Calculate well-level statistics (mean, median, std)
   - Generate organoid-parent aggregations
   - Compute patient-level summaries

6. **Consensus Profiles**

   - Merge sc and organoid aggregations
   - Create hierarchical profile structure
   - Export final consensus matrices

7. **QC Filtering**

   - Apply image QC flags from Stage 1
   - Remove outlier objects (z-score > 3)
   - Filter low-quality wells

8. **Patient Combination**

   - Merge profiles across all patients
   - Apply global feature selection
   - Generate all-patient consensus profiles

**Output Levels:**

- **Single-cell:** One row per nucleus/cell
- **Organoid:** One row per organoid (aggregated from cells)
- **Well:** One row per well FOV (aggregated from organoids)
- **Patient:** One row per patient/treatment (aggregated from wells)

**Inputs:**

- Feature CSV files from Stage 3
- Metadata: plate maps, treatment info, QC flags

**Outputs:**

- ``data/{patient}/image_based_profiles/sc.parquet`` - Single-cell profiles
- ``data/{patient}/image_based_profiles/organoid.parquet`` - Organoid profiles
- ``data/all_patient_profiles/sc_consensus.parquet`` - Cross-patient SC
- ``data/all_patient_profiles/organoid_consensus.parquet`` - Cross-patient organoid
- ``data/all_patient_profiles/well_aggregated.parquet`` - Well-level
- ``data/all_patient_profiles/patient_aggregated.parquet`` - Patient-level

**Feature Selection Parameters:**

- Correlation threshold: 0.9
- Variance threshold: 0.01
- NA cutoff: 5%
- Frequency cut: 0.1
- Unique cut: 0.1

**Execution:**

.. code-block:: bash

   cd 4.processing_image_based_profiles
   sbatch merge_features_grand_parent.sh

Data Organization
==================================================

Directory Structure
--------------------------------------------------

The pipeline expects data organized in this hierarchy::

    NF1_3D_organoid_profiling_pipeline/
    ├── data/
    │   ├── patient_IDs.txt
    │   ├── NF0014_T1/
    │   │   ├── zstack_images/
    │   │   │   ├── C4-2/
    │   │   │   │   ├── 405.tif        # DNA channel
    │   │   │   │   ├── 488.tif        # ER channel
    │   │   │   │   ├── 555.tif        # Golgi channel
    │   │   │   │   ├── 568.tif        # F-actin channel
    │   │   │   │   └── 640.tif        # Mito channel
    │   │   │   └── ... (other well FOVs)
    │   │   ├── segmentation_masks/
    │   │   │   ├── C4-2/
    │   │   │   │   ├── organoid_mask.tif
    │   │   │   │   ├── nuclei_mask.tif
    │   │   │   │   ├── cell_mask.tif
    │   │   │   │   └── cytoplasm_derived.tif
    │   │   │   └── ...
    │   │   ├── extracted_features/
    │   │   │   ├── C4-2/
    │   │   │   │   ├── AreaSizeShape_Nuclei_DNA_CPU.parquet
    │   │   │   │   ├── Intensity_Cell_488_GPU.parquet
    │   │   │   │   ├── Texture_Cytoplasm_640_CPU.parquet
    │   │   │   │   └── ... (125-189 files)
    │   │   │   └── ...
    │   │   ├── image_based_profiles/
    │   │   │   ├── 0.converted_profiles/
    │   │   │   │   ├── C4-2/
    │   │   │   │   │   ├── sc_related.parquet
    │   │   │   │   │   └── organoid_related.parquet
    │   │   │   ├── 1.combined_profiles/
    │   │   │   │   ├── sc.parquet
    │   │   │   │   └── organoid.parquet
    │   │   │   ├── 2.annotated_profiles/
    │   │   │   ├── 3.normalized_profiles/
    │   │   │   ├── 4.feature_selected_profiles/
    │   │   │   └── 5.aggregated_profiles/
    │   │   └── qc_flags.parquet
    │   ├── NF0016_T1/
    │   │   └── ... (same structure)
    │   └── all_patient_profiles/
    │       ├── sc_consensus.parquet
    │       ├── organoid_consensus.parquet
    │       ├── well_aggregated.parquet
    │       └── patient_aggregated.parquet
    ├── models/
    │   └── sam-med3d-turbo.pth
    ├── environments/
    │   ├── GFF_preprocessing.yml
    │   ├── GFF_segmentation.yml
    │   └── ... (conda environments)
    └── ... (code directories 0-6)

File Naming Conventions
--------------------------------------------------

**Z-Stack Images:**

- Format: ``{channel}.tif`` where channel ∈ {405, 488, 555, 568, 640}
- Dimensions: (Z, Y, X)
- Data type: uint16

**Segmentation Masks:**

- Format: ``{compartment}_mask.tif``
- Compartments: {organoid, nuclei, cell, cytoplasm}
- Label encoding: Integer object IDs (0=background, 1-N=objects)

**Feature Files:**

- Format: ``{feature}_{compartment}_{channel}_{processor}_features.parquet``
- Example: ``Intensity_Nuclei_405_GPU_features.parquet``

**Profile Files:**

- Format: Parquet (compressed columnar storage)
- Naming: ``{level}_{aggregation}.parquet``
- Example: ``sc_consensus.parquet``

Channel Information
==================================================

The pipeline processes five fluorescent imaging channels:

.. table:: Channel Specifications
   :widths: 15 20 10 10 10 15 20

   +--------+----------------------+--------+--------+----------+----------------+-------------------+
   | Name   | Fluorophore          | Ex(nm) | Em(nm) | Dichroic | Target         | Organelle         |
   +========+======================+========+========+==========+================+===================+
   | 405    | Hoechst 33342        | 361    | 486    | 405      | DNA            | Nucleus           |
   +--------+----------------------+--------+--------+----------+----------------+-------------------+
   | 488    | ConA Alexa Fluor 488 | 495    | 519    | 488      | ER             | ER                |
   +--------+----------------------+--------+--------+----------+----------------+-------------------+
   | 555    | WGA Alexa Fluor 555  | 555    | 580    | 555      | Membranes      | Golgi/Plasma Memb |
   +--------+----------------------+--------+--------+----------+----------------+-------------------+
   | 568    | Phalloidin AF 568    | 578    | 600    | 555      | F-actin        | Cytoskeleton      |
   +--------+----------------------+--------+--------+----------+----------------+-------------------+
   | 640    | MitoTracker Deep Red | 644    | 665    | 640      | Mitochondria   | Mitochondria      |
   +--------+----------------------+--------+--------+----------+----------------+-------------------+

**Imaging Parameters:**

- Objective: 60x/1.35 NA oil immersion
- Oil RI: 1.518
- Voxel size: 0.108 µm (XY) × 1 µm (Z)
- Bit depth: 16-bit
- Dynamic range: 0-65535

Computational Requirements
==================================================

Hardware Specifications
--------------------------------------------------

**Local**

- CPU: 24 cores @ 2.5 GHz
- RAM: 128 GB
- Storage: 20 TB free space
- GPU: NVIDIA GeForce 3090Ti with 24 GB VRAM for acceleration

**HPC Cluster (SLURM):**

- Nodes: 100s of CPU compute nodes
- Partition: amilan (CPU), aa100 (GPU)
- QOS: normal (24h), long (7 days)
- Max concurrent jobs: 990 per user

Software Environment
--------------------------------------------------

**Operating System:**

- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (limited support)

**Conda Environments:**

- ``GFF_preprocessing``: Data ingestion and z-stack creation
- ``GFF_segmentation``: Cellpose, SAM-Med3D for segmentation
- ``GFF_DL_featurization``: Deep learning feature extraction
- ``GFF_cellprofiling``: CPU/GPU feature extraction
- ``GFF_EDA``: Python analysis (pandas, scikit-learn, seaborn)
- ``gff_figure_env``: R visualization (ggplot2, tidyverse)

**Key Dependencies:**

- Python 3.9-3.11
- PyTorch 2.0+ (with CUDA 11.8+)
- Cellpose 3.x, 4.x
- R 4.3+
- SLURM workload manager

Runtime Estimates
--------------------------------------------------

**Per-well fov Processing Time:**

.. table:: Stage Runtimes
   :widths: 30 20 30

   +----------------------------+----------+---------------------------+
   | Stage                      | CPU Time |  Notes                    |
   +============================+==========+===========================+
   | 0. Preprocessing           | 10 mins  | Depends on file I/O       |
   +----------------------------+----------+---------------------------+
   | 1. QC                      | <30 min  | CellProfiler analysis     |
   +----------------------------+----------+---------------------------+
   | 2. Segmentation            | 30 mins  | Cellpose on GPU           |
   +----------------------------+----------+---------------------------+
   | 3. Featurization           | 2 hrs    | GPU: Intensity, Coloc     |
   +----------------------------+----------+---------------------------+
   | 4. Profile Processing      | 10 mins  | Pandas aggregations       |
   +----------------------------+----------+---------------------------+
   | **Total (CPU only)**       | <3 hrs   |                           |
   +----------------------------+----------+---------------------------+
   | **Total (with GPU)**       | N/A      | **Recommended**           |
   +----------------------------+----------+---------------------------+

**Storage Requirements:**

- Raw images: 250-500 MB/well FOV
- Z-stacks: 250-500 MB/well FOV
- Masks: 250-500 MB/well FOV
- Features: 5-10 MB/well FOV
- Profiles: 1-5 MB/well FOV
- **Total: ~1-2 GB/well FOV**

Number of FOVs per well varies between 7-25 with typically 60 wells per patient.
Per patient well fovs can range from 420 to 1500 depending on the experiment design.
Estimations for storage per patient are as follows:

.. table:: Storage Estimates per Patient
    :widths: 20 20

    +----------------+----------------+
    | Well FOVs      |  Storage (TB)  |
    +================+================+
    | 400            |   ~0.4-0.8     |
    +----------------+----------------+
    | 500            |   ~0.5-1.0     |
    +----------------+----------------+
    | 1000           |   ~1.0-2.0     |
    +----------------+----------------+
    | 1500           |   ~1.5-3.0     |
    +----------------+----------------+

