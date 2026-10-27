# Featurization merging

The approach to the featurization is to run each feature extraction function for each cell compartment for each channel in a distributed manner.
The results are then combined into a single dataframe for each cell compartment and channel.
The final distinct features are saved as parquet files.
These parquet files are then merged by cell compartment into:

- Nuclei
- Cell
- Cytoplasm
- Organoid

These are stored as related tables in a sqlite database.
The database is then used to merge into a single-cell feature table using cytotable.
For a visual and simplified representation of the pipeline, see the figure below.
![Featurization pipeline](./diagram/featurization_strategy.png)

## Feature selection blocklist

Features that contain coordinates are removed during feature selection.
These features would be leaked data if used in a machine learning model.

## File and information flow diagram

```mermaid
flowchart TD
    A1[cellpainting images and segmentations]


    A1 -->|featurization| B[nuclei features ]
    A1 -->|featurization| C[cell features ]
    A1 -->|featurization| D[cytoplasm features ]
    A1 -->|featurization| E[organoid features ]
    A1 -->|featurization| F[nucleocentric features ]


    B --> |merging| G[single-cell features ]
    C --> |merging| G[single-cell features ]
    D --> |merging| G[single-cell features ]
    G --> |annotation| G1[single-cell features ]

    E --> |annotation| H[organoid features ]
    F --> |annotation| I[nucleocentric features ]
    G1 --> J[single-cell handcrafted features ]
    G1 --> K[single-cell deep learning features ]
    H --> L[organoid handcrafted features]
    H --> M[organoid deep learning features]
    I --> N[nucleocentric volumetric features]
    I --> O[nucleocentric flat features]
    J --> |QC| P1[QC profiles]
    K --> |QC| P2[QC profiles]
    L --> |QC| P3[QC profiles]
    M --> |QC| P4[QC profiles]
    N --> |QC| P5[QC profiles]
    O --> |QC| P6[QC profiles]
    P1 --> |normalization| S1[normalized profiles]
    P2 --> |normalization| S2[normalized profiles]
    P3 --> |normalization| S3[normalized profiles]
    P4 --> |normalization| S4[normalized profiles]
    P5 --> |normalization| S5[normalized profiles]
    P6 --> |normalization| S6[normalized profiles]
    S1 --> |feature selection| T1[selected features]
    S2 --> |feature selection| T2[selected features]
    S3 --> |feature selection| T3[selected features]
    S4 --> |feature selection| T4[selected features]
    S5 --> |feature selection| T5[selected features]
    S6 --> |feature selection| T6[selected features]
    T1 --> U1[aggregated profiles]
    T2 --> U2[aggregated profiles]
    T3 --> U3[aggregated profiles]
    T4 --> U4[aggregated profiles]
    T5 --> U5[aggregated profiles]
    T6 --> U6[aggregated profiles]
    T1 --> V1[consensus profiles]
    T2 --> V2[consensus profiles]
    T3 --> V3[consensus profiles]
    T4 --> V4[consensus profiles]
    T5 --> V5[consensus profiles]
    T6 --> V6[consensus profiles]
```

## Number of feature files per image-set (well_fov)

| Feature type     | # of compartments | # of channels | Total number of feature files |
| ---------------- | ----------------- | ------------- | ----------------------------- |
| AreaSizeShape    | 4                 | 1             | 4                             |
| Colocalization   | 4                 | 6             | 24                            |
| Intensity        | 4                 | 4             | 16                            |
| Granularity      | 4                 | 4             | 16                            |
| Neighbors        | 1                 | 1             | 1                             |
| Texture          | 4                 | 4             | 16                            |
| Deep learning    | 4                 | 4             | 16                            |
| Nucleocentric 3D | 1                 | 4             | 4                             |
| Nucleocentric 2D | 1                 | 4             | 4                             |
| Total            |                   |               | 101                           |

## New flow of data given ZEDProfiler nextflow runs

```mermaid
flowchart TD
    A[ZEDProfiler feature warehouse] --> A1[2a.write_warehouse_views_to_parquet.ipynb]
    A1 --> B[3.organoid_cell_relationships.ipynb]
    C[DL features] --> D[1.merge_feature_parquets.ipynb]
    D --> E[2.merge_sc.ipynb]
    E --> B
    B --> F[5.combining_profiles.ipynb]
    F --> G[6.annotation.ipynb]
    G --> H[7a.organoid_qc.ipynb]
    H --> I[7b.single_cell_qc.ipynb]
    I --> J[7c.propagate_cqc_to_dl_profiles.ipynb]
    J --> K[8.normalization.ipynb]
    K --> L[9.feature_selection.ipynb]
    L --> M[10.aggregation.ipynb]
    M --> N[11.combine_patients.ipynb]
    N --> O[12.validate_profiles.ipynb]
```

## Data structure

The input data are located at:

`~/mnt/bandicoot/NF1_organoid_data/data`

The directory is organized by processing stage and profile type showing one patient as an example.

```text
├── NF0014_T1
│   ├── extracted_features
│   ├── image_based_profiles
│   │   ├── 0.converted_profiles
│   │   ├── 1.related_profiles
│   │   ├── 2.combined_profiles
│   │   ├── 3.annotated_profiles
│   │   ├── 4.qc_profiles
│   │   ├── 5.normalized_profiles
│   │   ├── 6.feature_selected_profiles
│   │   ├── 7.aggregated_profiles
│   │   └── 8.consensus_profiles
│   ├── zstack_images
│   └── segmentation_masks
├── NF0014_T2
├── NF0016_T1
├── NF0018_T6
├── NF0021_T1
├── NF0030_T1
├── NF0035_T1
├── NF0037_T1
├── NF0037_T1_CQ1
├── NF0040_T1
├── NF0055_T1
├── SARCO219_T2
└── SARCO361_T1
```

notebookes 1-2 write files to the dir:

```
├── image_based_profiles
│   ├── 0.converted_profiles
```

Notebook `3.organoid_cell_relationships.ipynb` reads from the above dir:
and continues through the ibp pipeline.
