#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=00:5:00
#SBATCH --output=z-norm-%j.out
#SBATCH --array=1-105


module load anaconda

conda activate gff_preprocessing_env

# get the current directory
# get a list of all directories in the data/z-stack_images directory

dirs_list=$(ls -d ../../data/z-stack_images/*)
# make the list of directories into an array
dirs_list=($dirs_list)
n_sets=${#dirs_list[@]}
echo $n_sets

# index the list of directories
job_id=$((SLURM_ARRAY_TASK_ID - 1))
dir_idx=$((job_id % ${#dirs_list[@]}))

dir=${dirs_list[$dir_idx]}
echo "$dir"
python 2.z_slice_instensity_normalization.py --input_dir "$dir"

conda deactivate

echo "Z-slice intensity normalization complete"
