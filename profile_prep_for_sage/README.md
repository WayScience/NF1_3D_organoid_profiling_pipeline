# NF1 patient derived organoid image-based profiling data

This README explains what the files in this directory are and provides background on how the data were extracted.

## Background

This directory contains the results of the image-based profiling of patient-derived organoids.
The data were extracted using a custom computer vision pipeline designed to analyze images of organoids.
The profiling includes various features extracted from the images, which are stored in different formats for further analysis.

### Image based profiles

An image-based profile is a set of features extracted from objects (e.g., single nuclei, cells, cytoplasm, organoids) in images; where an image is a single channel representing a stained structure (e.g., nuclei, mitochondria, golgi, etc.).
These features can include measurements such as areasizeshape, colocalization, granularity, intensity, texture, and others.

## The data structure

The data are organized into several files, each containing different types of profiles.
The profiles are set up with features as the columns and the objects (e.g., organoids, single cells) as the rows.
The files are in parquet format, which is a columnar storage file format optimized for use with big data processing frameworks like apache spark and pandas.
This format allows for efficient storage and retrieval of large datasets, making it suitable for the profiling data.

### File descriptions

Prior to describing the files, it is important to note that the files are named according to the type of profile they contain.

#### Compartments

- Organoid profiles are those that contain features extracted from organoids.
- Single cell (sc) profiles contain features extracted from single cells.
  The single cell profiles contain features extracted from the nuclear, cell, and cytoplasmic compartments of the cells.
- Nulceocentric profiles are profiles extracted from crops around the nuclei of the cells.

Both the organoid and single cell profiles are further divided into different types of profiles based on the type of feature extraction performed.

#### Feature extraction types

- Handcrafted profiles are the handcrafted profiles. these are the features extracted from the images using the custom computer vision pipeline.
- SAMMed3D profiles are the SAM-segmentation-derived profiles. these are the features extracted from the images using SAM.
  - SAMMed3d gets used in two ways: 1) masked featurizationand 2) cropped featurization.
- Moprhem profiles are the nucleocentric/MorphEM-derived profiles. these are the features extracted from the images using MorphEM.

- Normalized profiles are normalized to the negative control (DMSO) wells on a per-plate basis using robust z-scoring.
- Feature selected profiles are the feature selected profiles. we use pycytominer to feature select our profiles. if you would like non-feature selected profiles, please request.
  - Note that currently these profiles are normalized at the patient and tumor level, while feature selection is performed across all patients currently. please use the feature selected profiles with this in mind.
    All profiles after feature selection are based on the feature selected profiles.
- Aggregated profiles are the aggregated profiles - we perform median aggregation across all objects within a well (essentially making these profiles well-level profiles).
- Consensus profiles are the consensus profiles - we perform consensus aggregation across all objects within the same treatment condition or accross all replicates (essentially making these profiles treatment-level profiles).

The profile processing order will probably change in the future to have a more robust feature selection and normalization strategy.
The reason for feature selection being performed on all patients is so that the same features are selected across all patients, allowing for easier comparison.
Of course depending on the down stream task this might not actually matter.

Profiles are also generated separately for each z-projection strategy used to flatten the 3D image stacks:

- `middle slice` - profiles extracted from a single slice around the middle of the z-stack.
- `middle_n_slice` - profiles extracted from a small number of slices around the middle of the z-stack.
- `max_projection` - profiles extracted from a maximum-intensity projection of the z-stack.

## File metadata in the name

Each profile type is stored as a directory named `{profile_type}.parquet/`, which contains one parquet file per patient/tumor, treatment, and dose combination:

`{profile_type}.parquet/{PatientTumor}_{Treatment}_{Dose}_{Unit}.parquet`

Where any replicates for the same patient/tumor, treatment, and dose (with its unit) are all stored in one file.

`profile_type` reflects the object type (`organoid` or `sc` for single-cell) and the processing step (e.g. `profiles`, `fs` for feature selected, `agg` for aggregated, `consensus`, `norm_fs` for normalized + feature selected). Feature selected profiles under `1.feature_selected_profiles/` may additionally be prefixed with `sammed_` (SAM-segmentation-derived) or contain `nucleocentric_morphem` to indicate the nucleocentric/MorphEM feature set they were derived from.
