#!/bin/bash

conda activate gff_preprocessing_env
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/0.preprocessing_data/scripts/ "$git_root"/0.preprocessing_data/notebooks/*.ipynb

python "$git_root"/0.preprocessing_data/scripts/0.patient_specific_preprocessing.py
python "$git_root"/0.preprocessing_data/scripts/1.update_file_structure.py
python "$git_root"/0.preprocessing_data/scripts/2.make_z-stack_images.py


conda deactivate

echo "Preprocessing complete"

