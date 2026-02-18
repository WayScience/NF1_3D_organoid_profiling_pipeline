#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=4:00:00
#SBATCH --output=multi_process_2D_3D_tech_analysis_%j.out

module load anaconda
conda init bash
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

conda activate GFF_segmentation

python raw_image_tech_analysis_2D_3D.py --num_processes 128

conda deactivate

cd ../ || exit

echo "2D and 3D technical analysis complete"
