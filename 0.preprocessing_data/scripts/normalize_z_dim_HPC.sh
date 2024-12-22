#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=00:60:00
#SBATCH --output=preprocessing-%j.out
#SBATCH --array=1-2


module load anaconda

conda activate gff_preprocessing_env

# get the current directory
# get a list of all directories in the data/z-stack_images directory

dirs_list=$(ls -d ../../data/z-stack_images/*)

# index the list of directories
job_id=$SLURM_ARRAY_TASK_ID
dir_idx=$((job_id % ${#dirs_list[@]}))

dir=${dirs_list[$dir_idx]}

python 2.z_slice_instensity_normalization.py --input_dir $dir

conda deactivate

echo "Z-slice intensity normalization complete"
