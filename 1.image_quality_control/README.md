# Whole image quality control

Acquisition of Cell Painting images is not a perfect science, and there is always at least one image that is of poor quality.
Poor quality can be defined as:

- Large smudge/artifact within FOV (can take up part of or majority of the image)
- Debris scattered across the FOV
- Out-of-focus/blurry FOV
- Over-saturated cells (could be due to high concentration of stain)
- Empty FOV

In this project, we have two different "types" of datasets:

1. **Pipeline optimization datasets:** Plate(s) of data from patients being used for optimizing the image-based profiling pipeline (e.g., segmentation, feature extraction, etc.).
2. **Channel condition optimization datasets:** Plate(s) of data that include various conditions (e.g., different LED power settings and exposure times) for specific channels to determine which is most optimal to capture the organoid(s).

We use CellProfiler to extract blur (measurement=PowerLogLogSlope) and over-saturation (measurement=PercentMaximal) metrics per z-slice per channel.
There are six notebooks in this module:

0. Run CellProfiler QC pipeline to extract QC metrics and export as CSV files.
1. Evaluate blur metrics for the pipeline optimization dataset(s)
2. Evaluate saturation metrics for the pipeline optimization dataset(s)
3. Generate a QC report for the pipeline optimization dataset(s), which includes plots generated using ggplot
4. Evaluate blur and saturation metrics for the channel condition optimization dataset(s)
5. Generate a QC report for the channel condition optimization dataset(s), which includes plots generated using ggplot

## Run whole image QC workflow

To run the whole image QC workflow, run the below command with the [run_image_qc.sh bash script](./run_image_qc.sh).

```bash
source run_image_qc.sh
```
