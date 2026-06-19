# Feature naming convention and schema specification

**Status:** experimental
**Version:** 0.0.1
**Date:** february 13, 2026
**Authors:** NF1 3D organoid profiling pipeline development team
**Keywords:** feature naming, schema, morphology, image analysis, 3D organoids

---

## Abstract

This document specifies the naming convention and schema for morphological features extracted from 3D organoid imaging data in the NF1 3D organoid profiling pipeline. the specification defines requirements for feature identifiers, data structures, and formatting rules to ensure consistency, interoperability, and maintainability across the analysis pipeline.

---

## 1. Introduction

### 1.1 Purpose

This specification establishes a standardized feature naming convention and data schema for 3D organoid image analysis. standardization enables:

- Consistent feature identification across analysis stages
- Automated feature parsing and metadata extraction
- Integration with downstream analysis tools
- Reproducible research outputs

### 1.2 Scope

This specification applies to all feature extraction modules within the pipeline, including but not limited to:

- Area, size, and shape measurements
- Colocalization analysis
- Granularity features
- Intensity measurements
- Neighbor relationships
- Deep learning features (sammed3d)
- Texture features

### 1.3 Key words

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## 2A. feature name format specification

### 2.1 General structure

Feature names MUST conform to the following structure:

```
<Compartment>_<channel>_<featuretype>_<measurement>
```

Where each component is separated by a single underscore character (`_`).

### 2.2 Component definitions

#### 2.2.1 Compartment component

The `<compartment>` component:

- MUST identify the cellular or spatial compartment from which the feature is extracted
- MUST be one of the following enumerated values:
  - `Nuclei` - nuclear compartment
  - `Cell` - whole cell compartment
  - `Cytoplasm` - cytoplasmic compartment (cell excluding nucleus)
  - `Organoid` - organoid-level compartment
- MUST NOT contain whitespace or special characters
- MUST use pascalcase capitalization

**Example:** `nuclei`, `cytoplasm`, `organoid`

#### 2.2.2 Channel component

The `<channel>` component:

- MUST identify the imaging channel or fluorophore used for the measurement
- MUST be one of the following values:
  - `DNA` - DAPI/hoechst nuclear stain (405nm excitation)
  - `AGP` - AGP marker (488nm excitation)
  - `ER` - endoplasmic reticulum marker (555nm excitation)
  - `Mito` - mitochondrial marker (640nm excitation)
  - `BF` - brightfield/transmitted light
- MUST NOT contain whitespace
- MAY use pascalcase capitalization
- MAY use hyphen-separated channel combinations for colocalization features (e.g., `DNA-mito`)
- MUST list channels in alphabetical order when combined (e.g., `DNA-mito` not `mito-DNA`)
- MUST be set to `nochannel` for channel-independent features (e.g., areasizeshape)

**Example:** `DNA`, `mito`, `DNA-mito`

#### 2.2.3 Featuretype component

The `<featuretype>` component:

- MUST identify the category or method of feature extraction
- MUST be one of the following enumerated values:
  - `Areasizeshape` - morphological measurements (area, volume, shape descriptors)
  - `Colocalization` - channel colocalization metrics
  - `Granularity` - granular spectrum and texture-at-scale features
  - `Intensity` - pixel intensity statistics
  - `Neighbors` - spatial relationship and neighbor counting
  - `Sammed3d` - deep learning features from SAM-med3d model
  - `Texture` - haralick texture features
- MUST NOT contain whitespace
- MUST use pascalcase capitalization
- MUST NOT include version numbers or implementation details

**Example:** `intensity`, `texture`, `colocalization`

#### 2.2.4 Measurement component

The `<measurement>` component:

- MUST identify the specific measurement or metric
- MUST NOT contain underscores, periods, spaces, or forward slashes
- MUST replace prohibited characters with hyphens (`-`)
- SHOULD use pascalcase for measurement names to maintain consistency
- MAY include parameter values appended with hyphens (e.g., `entropy-256-3`)
- MUST be descriptive and unambiguous

**Character replacement rules:**

- Underscore (`_`) → hyphen (`-`)
- Period (`.`) → hyphen (`-`)
- Space (` `) → hyphen (`-`)
- Forward slash (`/`) → hyphen (`-`)

**Example:** `meanintensity`, `entropy-256-3`, `angularsecondmoment`

### 2.3 Complete feature name examples

Valid feature names conforming to this specification:

```
Nuclei_DNA_Intensity_MeanIntensity
Cytoplasm_Mito_Texture_Entropy-256-3
Cell_DNA-Mito_Colocalization_Correlation
Organoid_NoChannel_AreaSizeShape_Volume
Nuclei_NoChannel_Neighbors_AdjacentCount
Cell_Mito_Granularity_Spectrum-10
Nuclei_DNA_SAMMed3D_CLSFeature-512
```

---

## 2B. metadata naming convention

### 2.1 General structure

Metadata are non morphology feature values that provide contextual information about the sample, experiment, or imaging conditions, or objects in the dataset.
Metadata are used to capture information that may be relevant for analysis, interpretation, or downstream processing but do not represent morphological measurements of the objects themselves.

Metadata names MUST conform to the following structure:

```
Metadata_<featurecategory>_<featurename>
```

Where each component is separated by a single underscore character (`_`).
The `Metadata_` prefix is used to clearly distinguish metadata features from morphological features in the dataset.
Each category metadata name MUST be in pascalcase and MUST NOT contain whitespace or special characters. the `<featurename>` component MUST be descriptive and unambiguous, following the same character restrictions as morphological feature names.

### 2.2 Metadata category definitions

The `<featurecategory>` component MUST identify the type (category) of metadata and MUST be one of the following enumerated values:

- `Storage` - metadata related to data storage and file management.
- `Biology` - metadata related to biological characteristics of the sample.
- `Experiment` - metadata related to experimental conditions and treatments.
- `Imaging` - metadata related to imaging parameters and conditions.
- `Microscopy` - metadata related to microscopy settings and configurations.
- `Object` - metadata related to specific objects (e.g., nuclei) or regions of interest in the dataset.
- `Neighbors` - metadata related to spatial relationships and neighbor counts of objects in the dataset.
- `Location` - metadata related to spatial information and coordinates.
- `Other` - this is a place holder for any metadata that might be used in the future that does not fit into the above categories. new categories can be added as needed, but the `other` category provides a catch-all for any metadata that does not fit into the predefined categories.

### 2.3 Complete metadata name examples

Valid metadata names conforming to this specification:

```
Metadata_Storage_FilePath
Metadata_Biology_PatientID
Metadata_Experiment_Treatment
Metadata_Imaging_ExposureTime
Metadata_Microscopy_Magnification
Metadata_Object_ObjectID
Metadata_Neighbors_AdjacentCount
Metadata_Location_Cell_CentroidX
```

## 3. References

### 3.1 Normative references

- **RFC 2119**: key words for use in rfcs to indicate requirement levels
  https://www.ietf.org/rfc/rfc2119.txt

- **Apache parquet format specification**
  https://parquet.apache.org/docs/file-format/

### 3.2 Informative references

- **Cellprofiler feature naming convention**
  Influenced naming structure for biological image analysis

- **OME data model**
  Open microscopy environment standards for microscopy data

---

## Copyright notice

Copyright (c) 2026 way science lab. all rights reserved.

This document may be freely distributed and used for implementation purposes within the NF1 3D organoid profiling pipeline project and related research activities. the license for this document is the covered under the license of the NF1 3D organoid profiling pipeline project, which is available at [LICENSE](../LICENSE).

---

**END OF SPECIFICATION**
