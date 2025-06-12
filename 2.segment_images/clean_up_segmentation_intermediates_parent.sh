#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=segmentation_grandparent-%j.out

# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "SACRO219" "SARCO361" )

cd scripts/ || exit

for patient in "${patient_array[@]}"; do
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    echo "Cleaning up segmentation files for patient: $patient"
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan\
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=segmentation_child-%j.out \
        clean_up_segmentation_intermediates_child.sh "$patient"

done

cd ../ || exit

conda deactivate
echo "Segmentation cleanup completed"
