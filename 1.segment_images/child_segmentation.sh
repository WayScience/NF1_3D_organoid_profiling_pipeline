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
module load anaconda
conda init bash
conda activate GFF_segmentation

cd scripts/ || exit

well_fov=$1
compartments=( "nuclei" "cell" "organoid" )

echo -ne "Processing directory $current_dir of $total_dirs\r"
echo "Reslicing images"
python 00.reslice_images.py --well_fov "$well_fov"
echo "Segmenting Nuclei"
python 0.segment_nuclei_organoids.py --well_fov "$well_fov" --window_size 2 --clip_limit 0.05 >> segmentation.log
echo "Completed Nuclei Segmentation"
echo "Segmenting Cells"
python 1.segment_cells_organoids.py --well_fov "$well_fov" --window_size 3 --clip_limit 0.1 >> segmentation.log
echo "Completed Cell Segmentation"
echo "Segmenting Organoids"
python 2.segment_whole_organoids.py --well_fov "$well_fov" --window_size 4 --clip_limit 0.1 >> segmentation.log
echo "Completed Organoid Segmentation"
for compartment in "${compartments[@]}"; do
    echo "Decoupling $compartment"
    python 3.segmentation_decoupling.py --well_fov "$well_fov" --compartment "$compartment" >> segmentation.log
    python 4.reconstruct_3D_masks.py --well_fov "$well_fov" --compartment "$compartment" --radius_constraint 10 >> segmentation.log
done
python 5.create_cytoplasm_masks.py --well_fov "$well_fov" >> segmentation.log




cd ../ || exit


# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
