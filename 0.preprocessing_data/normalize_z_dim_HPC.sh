#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=00:60:00
#SBATCH --output=preprocessing-%j.out
#SBATCH --array=1-3


module load anaconda

conda activate gff_preprocessing_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# get the current directory
# get a list of all directories in the data/z-stack_images directory

dirs_list=$(ls -d ../../data/z-stack_images/*)

# get the length of the list
len=${#dirs_list[@]}

# set up the job array
dir=${dirs_list[$SLURM_ARRAY_TASK_ID]}
echo "Processing directory: $dir"
python 2.z_slice_instensity_normalization.py --input_dir $dir

cd .. || exit

conda deactivate

echo "Z-slice intensity normalization complete"


