[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15028201.svg)](https://doi.org/10.5281/zenodo.15028201)

# pyroptosis_signature_data_analysis

Analysis Repository for the Interstellar project. For image analysis and image-based profiling for this project, please see [pyroptosis_signature_image_profiling](https://github.com/WayScience/pyroptosis_signature_image_profiling)

### Workflow:

Cells were incubated for 1h in the inhibitors, and then incubated for 6h in each inducer.
For more information on plate layout and experimental design please see [pyroptosis_signature_image_profiling](https://github.com/WayScience/pyroptosis_signature_image_profiling).

# Treatments of Interest (All in this table are with DMSO 0.025% inhibitor)

| Inducer         | Number of Doses | Dose Concentration(s)                             | Function                    |
| :-------------- | :-------------: | :------------------------------------------------ | :-------------------------- |
| DMSO            |        1        | 0.025%                                            | Vehicle Control             |
| Thapsigargin    |        2        | 1uM <br> 10uM                                     | ER Stress Induced Apoptosis |
| H202            |        2        | 100nM <br> 100uM                                  | Cell Death                  |
| Flagellin       |        2        | 0.1ug/mL <br> 1.0ug/mL                            | Induce Pyroptosis           |
| LPS             |        3        | 0.01ug/mL <br> 1.0ug/mL <br> 100ug/mL             | Induce Pyroptosis           |
| LPS + Nigericin |  1 \* 3 = (3)   | 1ug/mL + 1uM <br> 1ug/mL + 3uM <br> 1ug/mL + 10uM | Induce Pyroptosis           |

# Other Treatments and combinations

| Inducers       |  # of Doses  | Dose Concentration(s)                                                                                                | Inhibitors | # of Doses | Dose Concentration(s)     | Function                                   |
| :------------- | :----------: | :------------------------------------------------------------------------------------------------------------------- | :--------- | :--------: | :------------------------ | :----------------------------------------- |
| Media          |      1       | 0                                                                                                                    | Media      |     1      | 0                         | Control                                    |
| DMSO           |      1       | 0.0250%                                                                                                              | DMSO       |     2      | 0.025% <br> 1.0%          | Vehicle Control                            |
| DMSO           |      1       | 0.0250%                                                                                                              | Z-VAD-FMK  |     2      | 30uM <br> 100 uM          | Inhibit Inflamasome                        |
| Thapsigargin   |      2       | 1uM <br> 10uM                                                                                                        | DMSO       |     1      | 0.025%                    | ER Stress Induced Apoptosis                |
| H202           |      2       | 100nM <br> 100uM                                                                                                     | DMSO       |     1      | 0.025%                    | Cell Death                                 |
| H202           |      2       | 100nM <br> 100uM                                                                                                     | Z-VAD-FMK  |     1      | 100uM                     | Inhibit Inflamasome                        |
| H202           |      2       | 100nM <br> 100uM                                                                                                     | Disulfiram |     1      | 1uM                       | Cell Death                                 |
| Flagellin      |      2       | 0.1ug/mL <br> 1ug/mL                                                                                                 | DMSO       |     1      | 0.025%                    | Induce Pyroptosis                          |
| Flagellin      |      2       | 0.1ug/mL <br> 1ug/mL                                                                                                 | Disulfiram |     1      | 1uM                       | Induce Pyroptosis <br> Inhibit Inflamasome |
| LPS            |      5       | 0.01ug/mL <br> 0.1ug/mL <br> 1.0ug/mL <br> 10.0ug/mL <br> 100.0 ug/mL                                                | DMSO       |     1      | 0.025%                    | Induce Pyroptosis                          |
| LPS            |      1       | 10.0ug/mL                                                                                                            | Z-VAD-FMK  |     1      | 100uM                     | Induce Pyroptosis / Inhibit Inflamasome    |
| LPS            |      1       | 10.0ug/mL                                                                                                            | Disulfiram |     3      | 0.1uM <br> 1uM <br> 2.5uM | Induce Pyroptosis / Inhibit Inflamasome    |
| LPS + Nigercin | 2 \* 3 = (6) | 1ug/mL+1ug/mL <br> 1ug/mL+3ug/mL <br> 1ug/mL+10ug/mL <br> 100ug/mL+1ug/mL <br> 100ug/mL+3ug/mL <br> 100ug/mL+10ug/mL | DMSO       |     1      | 0.025%                    | Induce Pyroptosis                          |
| LPS + Nigercin | 1 \* 1 = (1) | 1ug/mL+10ug/mL                                                                                                       | Z-VAD-FMK  |     1      | 100uM                     | Induce Pyroptosis <br> Inhibit Inflamasome |
| LPS + Nigercin | 1 \* 1 = (1) | 1ug/mL+10ug/mL                                                                                                       | Disulfiram |     1      | 1uM                       | Induce Pyroptosis <br> Inhibit Inflamasome |
| Disulfiram     |      3       | 0.1uM <br> 1uM <br> 2.5uM                                                                                            | DMSO       |     1      | 0.025%                    | Pyroptosis Inhibitor                       |

| File name                                                | Cell line | Level                                        | Nomic data present? | Expected size |
| -------------------------------------------------------- | --------- | -------------------------------------------- | ------------------- | ------------- |
| PBMC_preprocessed_sc_norm_aggregated.parquet             | PBMC      | Normalized -> Aggregated                     | No                  | 2.9 MB        |
| PBMC_preprocessed_sc_norm_aggregated_nomic.parquet       | PBMC      | Normalized -> Aggregated                     | Yes                 | 3.3 MB        |
| PBMC_preprocessed_sc_norm_labeled.parquet                | PBMC      | Normalized -> Labeled                        | No                  | 78.4 GB       |
| PBMC_preprocess_sc_norm_no_fs.parquet                    | PBMC      | Normalized                                   | No                  | 165 GB        |
| PBMC_preprocess_sc_norm_no_fs_aggregated_nomic.parquet   | PBMC      | Normalized -> Aggregated                     | Yes                 | 3.3 MB        |
| PBMC_sc.parquet                                          | PBMC      | Raw                                          | No                  | 161.9 GB      |
| PBMC_sc_norm.parquet                                     | PBMC      | Normalized                                   | No                  | 165 GB        |
| PBMC_sc_norm_fs.parquet                                  | PBMC      | Normalized -> Feature selected               | No                  | 78.4 GB       |
| SHSY5Y_preprocessed_sc_norm_aggregated.parquet           | SHSY5Y    | Normalized -> Feature selected -> Aggregated | No                  | 3.0 MB        |
| SHSY5Y_preprocessed_sc_norm_aggregated_nomic.parquet     | SHSY5Y    | Normalized -> Feature selected -> Aggregated | Yes                 | 3.4 MB        |
| SHSY5Y_preprocessed_sc_norm_labeled.parquet              | SHSY5Y    | Normalized -> Labeled                        | No                  | 6.2 GB        |
| SHSY5Y_preprocess_sc_norm_no_fs.parquet                  | SHSY5Y    | Normalized                                   | No                  | 13.2 GB       |
| SHSY5Y_preprocess_sc_norm_no_fs_aggregated_nomic.parquet | SHSY5Y    | Normalized -> Aggregated                     | Yes                 | 6.8 MB        |
| SHSY5Y_sc.parquet                                        | SHSY5Y    | Raw                                          | No                  | 13.1 GB       |
| SHSY5Y_sc_norm.parquet                                   | SHSY5Y    | Normalized                                   | No                  | 13.3 GB       |
| SHSY5Y_sc_norm_fs.parquet                                | SHSY5Y    | Normalized -> Feature selected               | No                  | 6.2 GB        |
