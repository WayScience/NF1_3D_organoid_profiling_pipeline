#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=segmentation_grandparent-%j.out


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient_array=(
    "NF0014_T1"
    "NF0014_T2"
    "NF0016_T1"
    "NF0018_T6"
    "NF0021_T1"
    "NF0030_T1"
    "NF0031_T1"
    "NF0035_T1"
    "NF0037_T1-Z-1"
    "NF0037_T1-Z-0.5"
    "NF0037_T1-Z-0.2"
    "NF0037_T1-Z-0.1"
    "NF0040_T1"
    "SARCO219_T2"
    "SARCO361_T1"
)


for patient in "${patient_array[@]}"; do
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done

    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=long \
        --account=amc-general \
        --time=7-00:00:00 \
        --output=segmentation_parent-%j.out \
        "${git_root}"/2.segment_images/parent_segmentation.sh "$patient"

done


echo "All patients submitted for segmentation"
