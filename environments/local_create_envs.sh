#!/bin/bash

yaml_files=$(ls /*glob*.yml)

# read the first line of the yaml file
for yaml_file in $yaml_files; do
    # read the first line of the yaml file
    first_line=$(head -n 1 "$yaml_file")
    # parse the first line to get the environment name
    environment_name=$(echo "$first_line" | cut -d ' ' -f 2)
    # check if the environment exists
    if conda env list | grep -q "$environment_name"; then
        mamba env update -f "$yaml_file"
    else
        mamba env create -f "$yaml_file"
    fi
    # check if featurizer.yml is being created or...
    if [ "$yaml_file" == "featurizer.yml" ] || [ "$yaml_file" == "deep_learning_featurizer.yml" ]; then
        cd .. || exit 1
        conda activate "$environment_name"
        pip install -e ./src/featurization_utils
        cd environments || exit 1
        conda deactivate
    fi
    conda activate "$environment_name"
    pip install -e ./src/file_utils
    conda deactivate
done
