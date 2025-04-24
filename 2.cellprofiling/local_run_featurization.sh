#!/bin/bash

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

GPU_OPTIONS=( "TRUE" "FALSE" )

# start the timer
start_timestamp=$(date +%s)


parent_dir="../../data/NF0014/cellprofiler"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)

for dir in $dirs; do
    well_fov=$(basename $dir)
    for use_GPU in "${GPU_OPTIONS[@]}"; do
        echo $well_fov

        cd slurm_scripts || exit

        if [ "$use_GPU" = "TRUE" ]; then
            echo "Running GPU version"

            source run_area_shape_child.sh $well_fov $use_GPU

            source run_colocalization_child.sh $well_fov $use_GPU

            source run_granularity_child.sh $well_fov $use_GPU

            source run_intensity_child.sh $well_fov $use_GPU

        else
            echo "Running CPU version"

            source run_area_shape_child.sh $well_fov $use_GPU

            source run_colocalization_child.sh $well_fov $use_GPU

            source run_granularity_child.sh $well_fov $use_GPU

            source run_intensity_child.sh $well_fov $use_GPU
        fi
            source run_neighbors_child.sh $well_fov $use_GPU

            source run_texture_child.sh $well_fov $use_GPU
    done
done

cd ../ || exit

echo "Featurization done"
