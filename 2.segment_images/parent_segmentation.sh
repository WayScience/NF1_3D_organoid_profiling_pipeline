#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=2-00:00
#SBATCH --output=logs/parent/segmentation_parent-%j.out



jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


git_root=$(git rev-parse --show-toplevel)

if [ -d "/scratch/alpine" ]; then
    echo "Using Alpine environment"
    ENV_PATH="/projects/mlippincott@xsede.org/software/uv/envs/nf1_uv_env/.venv"
elif [ -d "/anvil" ]; then
    echo "Using Anvil environment"
    ENV_PATH="/anvil/projects/x-bio260064/software/uv/envs/nf1_uv_env/.venv"
else
    ENV_PATH="$git_root/.venv"
fi


PYTHON_BIN="$ENV_PATH/bin/python3"

"$PYTHON_BIN" "$git_root"/2.segment_images/scripts/get_run_combinations.py

txt_file="${git_root}/2.segment_images/load_data/load_combinations.txt"

# Check if TXT file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: TXT file not found at $txt_file"
    exit 1
fi

while IFS= read -r line; do
    # skip the header line
    if [[ "$line" == "patient"* ]]; then
        continue
    fi

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    input_subparent_name="${parts[2]}"
    mask_subparent_name="${parts[3]}"

    echo "Patient: $patient, WellFOV: $well_fov,  Input Subparent Name: $input_subparent_name, Mask Subparent Name: $mask_subparent_name"

    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done

    # requesting 2 nodes (3.75GB per node) for 7.5GB total memory requirement
    # --partition=aa100 \
    # --gres=gpu:1 \
    sbatch \
        --nodes=1 \
        --ntasks=6 \
        --partition=aa100 \
        --gres=gpu:1 \
        --qos=normal \
        --account=amc-general \
        --time=60:00 \
        --output=logs/child/segmentation_child-%j.out \
        "${git_root}"/2.segment_images/child_segmentation.sh "$patient" "$well_fov" "$input_subparent_name" "$mask_subparent_name"

done < "$txt_file"

echo "All segmentation child jobs submitted"

