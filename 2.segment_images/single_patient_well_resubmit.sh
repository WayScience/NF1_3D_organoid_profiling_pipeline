#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=10:00
#SBATCH --output=segmentation_resubmit-%j.out

# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

patient=$1

echo "Processing patient $patient"

# get all input directories in specified directory
rerun_dir="$git_root/2.segment_images/rerun_jobs.txt"
mapfile -t input_dirs < <(cat "$rerun_dir")
total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

# loop through all input directories
for well_fov in "${input_dirs[@]}"; do
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    well_fov=$(basename "$well_fov")
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    echo "Beginning segmentation for $well_fov"
    # requesting 6 nodes (3.75GB per node) for 22.5GB total memory requirement
    sbatch \
        --nodes=1 \
        --mem=30G \
        --partition=aa100 \
        --gres=gpu:1 \
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=segmentation_child-%j.out \
        "${git_root}"/2.segment_images/child_segmentation.sh "$patient" "$well_fov" "zstack_images" "segmentation_masks"
done

# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"
