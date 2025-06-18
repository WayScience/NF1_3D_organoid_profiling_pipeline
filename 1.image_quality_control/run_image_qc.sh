#!/bin/bash

# initialize the correct shell for your machine to allow conda to work
conda init bash
# activate the CellProfiler environment
conda activate gff_cp_env

# navigate to the directory containing the Jupyter notebooks
cd notebooks

# convert Jupyter notebook to Python script
jupyter nbconvert --to script --output-dir=../nbconverted/ *.ipynb

# run script(s)
python ../nbconverted/0.cp_image_qc.py

echo "CellProfiler runs have finished!"

# deactivate the CellProfiler environment
conda deactivate
# activate the preprocessing environment
conda activate gff_preprocessing_env

# run evaluation scripts
python ../nbconverted/1.evaluate_blur_qc.py
python ../nbconverted/2.evaluate_saturation_qc.py
python ../nbconverted/4.evaluate_qc_optimizaion.py

# deactivate the CellProfiler environment
conda deactivate
# activate the R environment
conda activate gff_r_env

# run reporting scripts
Rscript ../nbconverted/3.generate_qc_report.r
Rscript ../nbconverted/5.generate_qc_optimization_report.r

# navigate back to the image_quality_control directory
cd ..

echo "Image quality control complete."
