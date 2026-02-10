#!/bin/bash
# Run preprocessing locally using the configured conda environment.

conda activate gff_preprocessing_env
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/0.preprocessing_data/scripts/ "$git_root"/0.preprocessing_data/notebooks/*.ipynb

python "$git_root"/0.preprocessing_data/scripts/0.patient_specific_preprocessing.py
python "$git_root"/0.preprocessing_data/scripts/1.make_zstack_and_copy_over.py
python "$git_root"/0.preprocessing_data/scripts/2.perform_file_corruption_checks.py

conda deactivate

echo "Preprocessing complete"
