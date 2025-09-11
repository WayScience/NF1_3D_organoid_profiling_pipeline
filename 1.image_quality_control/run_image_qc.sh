#!/bin/bash
git_root="$(git rev-parse --show-toplevel)"
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
# initialize the correct shell for your machine to allow conda to work
conda init bash
# activate the CellProfiler environment
conda activate gff_cp_env

# convert Jupyter notebook to Python script
jupyter nbconvert --to script --output-dir="$git_root/1.image_quality_control/nbconverted/" "$git_root/1.image_quality_control/notebooks/*.ipynb"

# run script(s)
python "$git_root/1.image_quality_control/scripts/0.cp_image_qc.py"

echo "CellProfiler runs have finished!"

# deactivate the CellProfiler environment
conda deactivate
# activate the preprocessing environment
conda activate gff_preprocessing_env

# run evaluation scripts
python "$git_root/1.image_quality_control/scripts/1.evaluate_blur_qc.py"
python "$git_root/1.image_quality_control/scripts/2.evaluate_saturation_qc.py"
python "$git_root/1.image_quality_control/scripts/4.evaluate_qc_optimization.py"

# deactivate the CellProfiler environment
conda deactivate
# activate the R environment
conda activate gff_r_env

# run reporting scripts
Rscript "$git_root/1.image_quality_control/scripts/3.generate_qc_report.r"
Rscript "$git_root/1.image_quality_control/scripts/5.generate_qc_optimization_report.r"

echo "Image quality control complete."
