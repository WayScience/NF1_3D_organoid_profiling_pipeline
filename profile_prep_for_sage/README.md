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

- Organoid profiles are those that contain features extracted from organoids.
- Single cell (sc) profiles contain features extracted from single cells.
  The single cell profiles contain features extracted from the nuclear, cell, and cytoplasmic compartments of the cells.

Both the organoid and single cell profiles are further divided into different types of profiles based on the type of feature extraction performed.

- Normalized profiles are normalized to the negative control (DMSO) wells on a per-plate basis using robust z-scoring.
- Feature selected profiles are the feature selected profiles. we use pycytominer to feature select our profiles. if you would like non-feature selected profiles, please request.
  - Note that currently these profiles are normalized at the patient and tumor level, while feature selection is performed across all patients currently. please use the feature selected profiles with this in mind.
    All profiles after feature selection are based on the feature selected profiles.
- Aggregated profiles are the aggregated profiles - we perform median aggregation across all objects within a well (essentially making these profiles well-level profiles).
- Consensus profiles are the consensus profiles - we perform consensus aggregation across all objects within the same treatment condition or accross all replicates (essentially making these profiles treatment-level profiles).

The profile processing order will probably change in the future to have a more robust feature selection and normalization strategy.
The reason for feature selection being performed on all patients is so that the same features are selected across all patients, allowing for easier comparison.
Of course depending on the down stream task this might not actually matter.

## File metadata in the name

`Profile_type/{patient}_{Tumor_number}_{treatment}_{concentration}_{Concentration_unit}.parquet`
Where any replicates for the same patient, tumor number, treatment, and concentration are all stored in one file.
