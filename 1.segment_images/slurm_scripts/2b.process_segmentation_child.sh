#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=child_output_pt2-%j.out


module load anaconda

conda activate GFF_segmentation

dir=$1

echo "$dir"

cd .. || exit

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts || exit

python 4.create_cytoplasm_masks.py --input_dir "$dir"
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi
python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "cytoplasm"
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi
python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "nuclei"
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi
python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "cell"
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi

python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "organoid"
# ensure that the exit code is 0
if [ $? -ne 0 ]; then
    echo "Segmentation failed"
    exit 1
fi

cd ../slurm_scripts/ || exit

conda deactivate
echo "Segmentation complete"
