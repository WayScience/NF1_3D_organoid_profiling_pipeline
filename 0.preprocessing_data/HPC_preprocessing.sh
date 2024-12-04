#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=00:60:00
#SBATCH --output=preprocessing-%j.out

module load anaconda

conda activate gff_preprocessing_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

#python 0.update_file_structure.py --HPC True
python 1.make_z-stack_images.py

cd .. || exit

conda deactivate

echo "Preprocessing complete"

