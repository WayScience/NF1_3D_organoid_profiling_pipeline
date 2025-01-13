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
python nbconverted/0.cp_image_qc.py
python nbconverted/1.evaluate_blur_qc.py
python nbconverted/2.evaluate_saturation_qc.py

# navigate back to the image_quality_control directory
cd ..

echo "Image quality control complete."
