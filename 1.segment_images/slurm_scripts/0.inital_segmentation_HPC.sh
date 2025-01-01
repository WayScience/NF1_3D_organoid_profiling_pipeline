#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=segmentation-%j.out

module load anaconda

conda activate GFF_segmentation

cd .. || exit

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit
# get all input directories in specified directory
z_stack_dir="../../data/z-stack_images/"
#z_stack_dir="../../data/test_dir/"

# Use mapfile to read the output of ls -d into an array
mapfile -t input_dirs < <(ls -d "$z_stack_dir"*)

total_dirs=${#input_dirs[@]}

# Print each path to ensure they are separate elements
for dir in "${input_dirs[@]}"; do
    echo "Directory: $dir"
done

current_dir=0

# loop through all input directories
for dir in "${input_dirs[@]}"; do
    dir=${dir%*/}
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    python 0.segment_nuclei_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.05
    python 1.segment_cells_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.1
done

cd ../ || exit

conda deactivate

echo "Inital segmentation complete"
