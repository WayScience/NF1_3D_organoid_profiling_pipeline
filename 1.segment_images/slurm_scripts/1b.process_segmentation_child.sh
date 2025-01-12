#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=child_output_pt1-%j.out


module load anaconda

conda activate GFF_segmentation

dir=$1
compartment=$2

echo "$dir"
echo "$compartment"

cd .. || exit

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

# capture exit code for each script
python 3.segmentation_decoupling.py --input_dir "$dir" --compartment "$compartment"
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi
python 4.reconstruct_3D_masks.py --input_dir "$dir" --compartment "$compartment" --radius_constraint 10
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi
cd ../slurm_scripts/ || exit

conda deactivate
echo "Segmentation complete"
