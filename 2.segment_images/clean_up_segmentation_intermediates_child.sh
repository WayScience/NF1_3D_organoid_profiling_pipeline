#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=segmentation_cleanup-%j.out

# activate segmentation environment
module load anaconda
conda init bash
conda activate GFF_segmentation

patient=$1

cd scripts/ || exit

echo "Cleaning up segmentation files for patient: $patient"
python 9.clean_up_segmentation.py --patient "$patient"

cd ../ || exit

conda deactivate
echo "Segmentation cleanup completed"
