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
The database is then used to merge into a single-cell feature table using CytoTable.
For a visual and simplified representation of the pipeline, see the figure below.
![Featurization pipeline](./diagram/featurization_strategy.png)

## Feature selection blocklist
Features that contain coordinates are removed during feature selection.
These features would be leaked data if used in a machine learning model.

## File and information flow diagram
```mermaid
graph TD
    A1[CellPainting Images and Segmentations]


    A1 -->|Featurization| B[Nuclei features ]
    A1 -->|Featurization| C[Cell features ]
    A1 -->|Featurization| D[Cytoplasm features ]
    A1 -->|Featurization| E[Organoid features ]
    A1 -->|Featurization| F[Nucleocentric features ]


    B --> |Merging| G[Single-cell features ]
    C --> |Merging| G[Single-cell features ]
    D --> |Merging| G[Single-cell features ]
    G --> |Annotation| G1[Single-cell features ]

    E --> |Annotation| H[Organoid features ]
    F --> |Annotation| I[Nucleocentric features ]
    G1 --> J[Single-cell handcrafted features ]
    G1 --> K[Single-cell deep learning features ]
    H --> L[Organoid handcrafted features]
    H --> M[Organoid deep learning features]
    I --> N[Nucleocentric Volumetric features]
    I --> O[Nucleocentric flat features]
    J --> |QC| P1[QC profiles]
    K --> |QC| P2[QC profiles]
    L --> |QC| P3[QC profiles]
    M --> |QC| P4[QC profiles]
    N --> |QC| P5[QC profiles]
    O --> |QC| P6[QC profiles]
    P1 --> |Normalization| S1[Normalized profiles]
    P2 --> |Normalization| S2[Normalized profiles]
    P3 --> |Normalization| S3[Normalized profiles]
    P4 --> |Normalization| S4[Normalized profiles]
    P5 --> |Normalization| S5[Normalized profiles]
    P6 --> |Normalization| S6[Normalized profiles]
    S1 --> |Feature selection| T1[Selected features]
    S2 --> |Feature selection| T2[Selected features]
    S3 --> |Feature selection| T3[Selected features]
    S4 --> |Feature selection| T4[Selected features]
    S5 --> |Feature selection| T5[Selected features]
    S6 --> |Feature selection| T6[Selected features]
    T1 --> U1[Aggregated profiles]
    T2 --> U2[Aggregated profiles]
    T3 --> U3[Aggregated profiles]
    T4 --> U4[Aggregated profiles]
    T5 --> U5[Aggregated profiles]
    T6 --> U6[Aggregated profiles]
    T1 --> V1[Consensus profiles]
    T2 --> V2[Consensus profiles]
    T3 --> V3[Consensus profiles]
    T4 --> V4[Consensus profiles]
    T5 --> V5[Consensus profiles]
    T6 --> V6[Consensus profiles]
```


















