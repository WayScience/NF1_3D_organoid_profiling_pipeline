#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
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
python 2.segmentation_decoupling.py --input_dir "$dir" --compartment "$compartment"
python 3.reconstruct_3D_masks.py --input_dir "$dir" --compartment "$compartment" --radius_constraint 10
python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "$compartment"
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "5.make_cell_segmentation_videos.py failed with exit code $exit_code"
    exit $exit_code
fi

cd ../slurm_scripts/ || exit

conda deactivate
echo "Segmentation complete"
