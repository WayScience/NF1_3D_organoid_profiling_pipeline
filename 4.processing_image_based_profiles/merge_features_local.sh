#!/bin/bash


conda activate nf1_image_based_profiling_env

start_time=$(date +%s)

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/4.processing_image_based_profiles/scripts/ "$git_root"/4.processing_image_based_profiles/notebooks/*.ipynb

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi
load_data_file_path="$git_root/4.processing_image_based_profiles/load_data/load_file.txt"

bandicoot_dir="$HOME/mnt/bandicoot/NF1_organoid_data"
if [[ ! -d "$bandicoot_dir" ]]; then
    profile_base_dir="$git_root/"
else
    # shellcheck disable=SC2034
    profile_base_dir="$bandicoot_dir"
fi

# setup the logs dir
if [ -d "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/" ]; then
    rm -rf "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/"
fi
mkdir -p "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/" # create the patients directory if it doesn't exist




while IFS= read -r line; do

    IFS=$'\t' read -r -a parts <<< "$line"
    patient="${parts[0]}"
    well_fov="${parts[1]}"

    echo "$patient - $well_fov"
    log_file="$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/${patient}_${well_fov}.log"
    touch "$log_file"  # create the log file if it doesn't exist
    {
        # python "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py --patient "$patient" --well_fov "$well_fov" --output_features_subparent_name "extracted_features" --image_based_profiles_subparent_name "image_based_profiles"
        # python "$git_root"/4.processing_image_based_profiles/scripts/2.merge_sc.py --patient "$patient" --well_fov "$well_fov" --output_features_subparent_name "extracted_features" --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py --patient "$patient" --well_fov "$well_fov" --output_features_subparent_name "extracted_features" --image_based_profiles_subparent_name "image_based_profiles"
    } >> "$log_file" 2>&1
done < "$load_data_file_path"

for patient in "${patient_array[@]}"; do
    echo "Processing patient: $patient"

    patient_log_file="$git_root/4.processing_image_based_profiles/logs/patients/${patient}.log"
    mkdir -p "$(dirname "$patient_log_file")"  # create the patients directory if it doesn't exist
    touch "$patient_log_file"  # create the patient log file if it doesn't exist
    {
        python "$git_root"/4.processing_image_based_profiles/scripts/5.combining_profiles.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/6.annotation.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/7a.organoid_qc.py --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/7b.single_cell_qc.py --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/8.normalization.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/9.feature_selection.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/10.aggregation.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
        python "$git_root"/4.processing_image_based_profiles/scripts/11.merge_consensus_profiles.py --patient "$patient" --image_based_profiles_subparent_name "image_based_profiles"
    } >> "$patient_log_file" 2>&1

done



conda activate nf1_image_based_profiling_env

python "$git_root"/4.processing_image_based_profiles/scripts/12.combine_patients.py --image_based_profiles_subparent_name "image_based_profiles"
python "$git_root"/4.processing_image_based_profiles/scripts/0a.get_profiling_stats.py --image_based_profiles_subparent_name "image_based_profiles"
conda deactivate
conda activate gff_figure_env
Rscript "$git_root"/4.processing_image_based_profiles/scripts/0b.plot_profiling_stats.r
conda deactivate

echo "All features merged for patients" "${patient_array[@]}"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $elapsed_time seconds"
echo "Total elapsed time: $((elapsed_time / 60)) minutes"
echo "Total elapsed time: $((elapsed_time / 3600)) hours"
