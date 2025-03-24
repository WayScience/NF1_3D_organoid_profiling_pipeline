#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=segmentation-%j.out


# activate  cellprofiler environment
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit

dir=$1
compartments=( "nuclei" "cell" "organoid" )

echo "Segmenting Nuclei"
python 0.segment_nuclei_organoids.py --input_dir "$dir" --window_size 2 --clip_limit 0.05 >> segmentation.log
echo "Completed Nuclei Segmentation"
echo "Segmenting Cells"
python 1.segment_cells_organoids.py --input_dir "$dir" --window_size 3 --clip_limit 0.1 >> segmentation.log
echo "Completed Cell Segmentation"
echo "Segmenting Organoids"
python 2.segment_whole_organoids.py --input_dir "$dir" --window_size 4 --clip_limit 0.1 >> segmentation.log
echo "Completed Organoid Segmentation"
for compartment in "${compartments[@]}"; do
    echo "Decoupling $compartment"
    python 3.segmentation_decoupling.py --input_dir "$dir" --compartment "$compartment" >> segmentation.log
    python 4.reconstruct_3D_masks.py --input_dir "$dir" --compartment "$compartment" --radius_constraint 10 >> segmentation.log
done
python 5.create_cytoplasm_masks.py --input_dir "$dir" >> segmentation.log


cd ../ || exit


# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
