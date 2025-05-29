#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing-%j.out

module load anaconda

conda activate gff_preprocessing_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python 0.patient_specific_preprocessing.py --HPC
python 1.update_file_structure.py --HPC
python 2.make_z-stack_images.py

cd .. || exit

conda deactivate

echo "Preprocessing complete"

