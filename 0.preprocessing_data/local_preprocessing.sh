#!/bin/bash

conda activate gff_preprocessing_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# python 0.update_file_structure.py --HPC False
# python 1.make_z-stack_images.py

mapfile -t dirs < <(ls -d ../../data/z-stack_images/*)

for dir in "${dirs[@]}"; do
    echo "Processing directory: $dir"
    python 2.z_slice_instensity_normalization.py --input_dir "$dir"
done

cd .. || exit

conda deactivate

echo "Preprocessing complete"
