#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

txt_file="${git_root}/2.segment_images/load_data/load_combinations.txt"

# Check if TXT file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: TXT file not found at $txt_file"
    exit 1
fi

# get the total number of lines in the TXT file
total_lines=$(wc -l < "$txt_file")


jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ "$git_root"/2.segment_images/notebooks/*.ipynb


line_counter=0
while IFS= read -r line; do
    # increment the line counter
    line_counter=$((line_counter + 1))

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

    echo "Processing $patient-$well_fov | $line_counter/$total_lines"

    # shellcheck disable=SC1091
    source child_segmentation.sh "$patient" "$well_fov" "$input_subparent_name" "$mask_subparent_name"

# done < "$txt_file"
done < <(tac "$txt_file")



echo "All segmentation child jobs ran"

