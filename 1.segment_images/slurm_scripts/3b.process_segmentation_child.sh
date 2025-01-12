#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=child_output_pt3-%j.out


module load anaconda

conda activate GFF_segmentation

cd ../scripts/ || exit

python 7.clean_up_segmentation.py
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi

cd ../ || exit

conda deactivate
echo "Segmentation complete"
