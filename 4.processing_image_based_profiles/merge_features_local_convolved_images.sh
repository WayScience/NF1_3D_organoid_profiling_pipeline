#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

start_time=$(date +%s)


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient_array=(
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    "NF0014_T1"
    )

well_fov_array=(
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"
    "C4-2"

    )
output_features_subparent_name_array=(
    "extracted_features"
    "convolution_1_extracted_features"
    "convolution_2_extracted_features"
    "convolution_3_extracted_features"
    "convolution_4_extracted_features"
    "convolution_5_extracted_features"
    "convolution_6_extracted_features"
    "convolution_7_extracted_features"
    "convolution_8_extracted_features"
    "convolution_9_extracted_features"
    "convolution_10_extracted_features"
    "convolution_11_extracted_features"
    "convolution_12_extracted_features"
    "convolution_13_extracted_features"
    "convolution_14_extracted_features"
    "convolution_15_extracted_features"
    "convolution_16_extracted_features"
    "convolution_17_extracted_features"
    "convolution_18_extracted_features"
    "convolution_19_extracted_features"
    "convolution_20_extracted_features"
    "convolution_21_extracted_features"
    "convolution_22_extracted_features"
    "convolution_23_extracted_features"
    "convolution_24_extracted_features"
    "convolution_25_extracted_features"
    "convolution_50_extracted_features"
    "convolution_75_extracted_features"
    "convolution_100_extracted_features"

    )
image_based_profiles_subparent_name_array=(
    "image_based_profiles"
    "convolution_1_image_based_profiles"
    "convolution_2_image_based_profiles"
    "convolution_3_image_based_profiles"
    "convolution_4_image_based_profiles"
    "convolution_5_image_based_profiles"
    "convolution_6_image_based_profiles"
    "convolution_7_image_based_profiles"
    "convolution_8_image_based_profiles"
    "convolution_9_image_based_profiles"
    "convolution_10_image_based_profiles"
    "convolution_11_image_based_profiles"
    "convolution_12_image_based_profiles"
    "convolution_13_image_based_profiles"
    "convolution_14_image_based_profiles"
    "convolution_15_image_based_profiles"
    "convolution_16_image_based_profiles"
    "convolution_17_image_based_profiles"
    "convolution_18_image_based_profiles"
    "convolution_19_image_based_profiles"
    "convolution_20_image_based_profiles"
    "convolution_21_image_based_profiles"
    "convolution_22_image_based_profiles"
    "convolution_23_image_based_profiles"
    "convolution_24_image_based_profiles"
    "convolution_25_image_based_profiles"
    "convolution_50_image_based_profiles"
    "convolution_75_image_based_profiles"
    "convolution_100_image_based_profiles"
    )

# setup the logs dir
if [ -d logs/patient_well_fovs/ ]; then
    rm -rf logs/patient_well_fovs/
fi
mkdir -p logs/patient_well_fovs/ # create the patients directory if it doesn't exist

cd scripts || { echo "Scripts directory not found! Exiting."; exit 1; }

for index in "${!patient_array[@]}"; do
    patient="${patient_array[index]}"
    well_fov="${well_fov_array[index]}"
    output_features_subparent_name="${output_features_subparent_name_array[index]}"
    image_based_profiles_subparent_name="${image_based_profiles_subparent_name_array[index]}"
    echo "Processing patient: $patient, well_fov: $well_fov, output_features_subparent_name: $output_features_subparent_name, image_based_profiles_subparent_name: $image_based_profiles_subparent_name"

        # check if the well fov is a run stats dir
        if [[ $well_fov == *"run_stats"* ]]; then
            continue
        fi
        echo "$patient - $well_fov"
        log_file="../logs/${patient}_${well_fov}_${output_features_subparent_name}.log"
        touch "$log_file"  # create the log file if it doesn't exist
        # {
        #     python \
        #         1.merge_feature_parquets.py \
        #         --patient "$patient" --well_fov "$well_fov" \
        #         --output_features_subparent_name "${output_features_subparent_name}" \
        #         --image_based_profiles_subparent_name "${image_based_profiles_subparent_name}"
        #     python \
        #         2.merge_sc.py \
        #     --patient "$patient" --well_fov "$well_fov"\
        #     --output_features_subparent_name "${output_features_subparent_name}" \
        #     --image_based_profiles_subparent_name "${image_based_profiles_subparent_name}"
        #     python \
        #         3.organoid_cell_relationship.py \
        #     --patient "$patient" --well_fov "$well_fov"\
        #     --output_features_subparent_name "${output_features_subparent_name}" \
        #     --image_based_profiles_subparent_name "${image_based_profiles_subparent_name}"
        # } >> "$log_file" 2>&1
    patient_log_file="../logs/patients/${patient}.log"
    mkdir -p "$(dirname "$patient_log_file")"  # create the patients directory if it doesn't exist
    touch "$patient_log_file"  # create the patient log file if it doesn't exist
    {
        python 6.combining_profiles.py --patient "$patient" --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
        python 7.annotation.py --patient "$patient" --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
        python 8.normalization.py --patient "$patient" --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
        python 9.feature_selection.py --patient "$patient" --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
        python 10.aggregation.py --patient "$patient" --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
        python 11.merge_consensus_profiles.py --patient "$patient" --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
    } >> "$patient_log_file" 2>&1

    python 5a.organoid_qc.py --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"
    python 5b.single_cell_qc.py --image_based_profiles_subparent_name "$image_based_profiles_subparent_name"

done

conda deactivate


cd ../ || { echo "Failed to return to root directory! Exiting."; exit 1; }

echo "All features merged for patients" "${patient_array[@]}"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $elapsed_time seconds"
echo "Total elapsed time: $((elapsed_time / 60)) minutes"
echo "Total elapsed time: $((elapsed_time / 3600)) hours"
