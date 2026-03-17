# Feature Naming Convention and Schema Specification

**Status:** Experimental
**Version:** 0.0.1
**Date:** February 13, 2026
**Authors:** NF1 3D Organoid Profiling Pipeline Development Team
**Keywords:** feature naming, schema, morphology, image analysis, 3D organoids

---

## Abstract

This document specifies the naming convention and schema for morphological features extracted from 3D organoid imaging data in the NF1 3D Organoid Profiling Pipeline. The specification defines requirements for feature identifiers, data structures, and formatting rules to ensure consistency, interoperability, and maintainability across the analysis pipeline.

---

## 1. Introduction

### 1.1 Purpose

This specification establishes a standardized feature naming convention and data schema for 3D organoid image analysis. Standardization enables:

- Consistent feature identification across analysis stages
- Automated feature parsing and metadata extraction
- Integration with downstream analysis tools
- Reproducible research outputs

### 1.2 Scope

This specification applies to all feature extraction modules within the pipeline, including but not limited to:

- Area, Size, and Shape measurements
- Colocalization analysis
- Granularity features
- Intensity measurements
- Neighbor relationships
- Deep learning features (SAMMed3D)
- Texture features

### 1.3 Key Words

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## 2a. Feature Name Format Specification

### 2.1 General Structure

Feature names MUST conform to the following structure:

```
<Compartment>_<Channel>_<FeatureType>_<Measurement>
```

Where each component is separated by a single underscore character (`_`).

### 2.2 Component Definitions

#### 2.2.1 Compartment Component

The `<Compartment>` component:

- MUST identify the cellular or spatial compartment from which the feature is extracted
- MUST be one of the following enumerated values:
  - `Nuclei` - Nuclear compartment
  - `Cell` - Whole cell compartment
  - `Cytoplasm` - Cytoplasmic compartment (cell excluding nucleus)
  - `Organoid` - Organoid-level compartment
- MUST NOT contain whitespace or special characters
- MUST use PascalCase capitalization

**Example:** `Nuclei`, `Cytoplasm`, `Organoid`

#### 2.2.2 Channel Component

The `<Channel>` component:

- MUST identify the imaging channel or fluorophore used for the measurement
- MUST be one of the following values:
  - `DNA` - DAPI/Hoechst nuclear stain (405nm excitation)
  - `AGP` - AGP marker (488nm excitation)
  - `ER` - Endoplasmic reticulum marker (555nm excitation)
  - `Mito` - Mitochondrial marker (640nm excitation)
  - `BF` - Brightfield/transmitted light
- MUST NOT contain whitespace
- MAY use PascalCase capitalization
- MAY use hyphen-separated channel combinations for colocalization features (e.g., `DNA-Mito`)
- MUST list channels in alphabetical order when combined (e.g., `DNA-Mito` not `Mito-DNA`)
- MUST be set to `NoChannel` for channel-independent features (e.g., AreaSizeShape)

**Example:** `DNA`, `Mito`, `DNA-Mito`

#### 2.2.3 FeatureType Component

The `<FeatureType>` component:

- MUST identify the category or method of feature extraction
- MUST be one of the following enumerated values:
  - `AreaSizeShape` - Morphological measurements (area, volume, shape descriptors)
  - `Colocalization` - Channel colocalization metrics
  - `Granularity` - Granular spectrum and texture-at-scale features
  - `Intensity` - Pixel intensity statistics
  - `Neighbors` - Spatial relationship and neighbor counting
  - `SAMMed3D` - Deep learning features from SAM-Med3D model
  - `Texture` - Haralick texture features
- MUST NOT contain whitespace
- MUST use PascalCase capitalization
- MUST NOT include version numbers or implementation details

**Example:** `Intensity`, `Texture`, `Colocalization`

#### 2.2.4 Measurement Component

The `<Measurement>` component:

- MUST identify the specific measurement or metric
- MUST NOT contain underscores, periods, spaces, or forward slashes
- MUST replace prohibited characters with hyphens (`-`)
- SHOULD use PascalCase for measurement names to maintain consistency
- MAY include parameter values appended with hyphens (e.g., `Entropy-256-3`)
- MUST be descriptive and unambiguous

**Character Replacement Rules:**
- Underscore (`_`) → Hyphen (`-`)
- Period (`.`) → Hyphen (`-`)
- Space (` `) → Hyphen (`-`)
- Forward slash (`/`) → Hyphen (`-`)

**Example:** `MeanIntensity`, `Entropy-256-3`, `AngularSecondMoment`

### 2.3 Complete Feature Name Examples

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

## 2b. Metadata Naming Convention


### 2.1 General Structure

Metadata are non morphology feature values that provide contextual information about the sample, experiment, or imaging conditions, or objects in the dataset.
Metadata are used to capture information that may be relevant for analysis, interpretation, or downstream processing but do not represent morphological measurements of the objects themselves.

Metadata names MUST conform to the following structure:

```
Metadata_<FeatureCategory>_<FeatureName>
```

Where each component is separated by a single underscore character (`_`).
The `Metadata_` prefix is used to clearly distinguish metadata features from morphological features in the dataset.
Each category metadata name MUST be in PascalCase and MUST NOT contain whitespace or special characters. The `<FeatureName>` component MUST be descriptive and unambiguous, following the same character restrictions as morphological feature names.

### 2.2 Metadata Category Definitions
The `<FeatureCategory>` component MUST identify the type (Category) of metadata and MUST be one of the following enumerated values:
- `Storage` - Metadata related to data storage and file management.
- `Biology` - Metadata related to biological characteristics of the sample.
- `Experiment` - Metadata related to experimental conditions and treatments.
- `Imaging` - Metadata related to imaging parameters and conditions.
- `Microscopy` - Metadata related to microscopy settings and configurations.
- `Object` - Metadata related to specific objects or regions of interest in the dataset.
- `Neighbors` - Metadata related to spatial relationships and neighbor counts of objects in the dataset.
- `Location` - Metadata related to spatial information and coordinates.

- `Other` - This is a place holder for any metadata that might be used in the future that does not fit into the above categories. New categories can be added as needed, but the `Other` category provides a catch-all for any metadata that does not fit into the predefined categories.

### 2.3 Complete Metadata Name Examples
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

### 3.1 Normative References

- **RFC 2119**: Key words for use in RFCs to Indicate Requirement Levels
  https://www.ietf.org/rfc/rfc2119.txt

- **Apache Parquet Format Specification**
  https://parquet.apache.org/docs/file-format/

### 3.2 Informative References

- **CellProfiler Feature Naming Convention**
  Influenced naming structure for biological image analysis

- **OME Data Model**
  Open Microscopy Environment standards for microscopy data

---

## Copyright Notice

Copyright (c) 2026 Way Science Lab. All rights reserved.

This document may be freely distributed and used for implementation purposes within the NF1 3D Organoid Profiling Pipeline project and related research activities. The license for this document is the covered under the license of the NF1 3D Organoid Profiling Pipeline project, which is available at [LICENSE](../LICENSE).

---

**END OF SPECIFICATION**
