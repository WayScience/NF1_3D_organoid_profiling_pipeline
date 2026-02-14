#!/bin/bash

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

conda activate GFF_segmentation

python raw_image_tech_analysis_2D_3D.py

conda deactivate

cd ../ || exit

echo "2D and 3D technical analysis complete"
