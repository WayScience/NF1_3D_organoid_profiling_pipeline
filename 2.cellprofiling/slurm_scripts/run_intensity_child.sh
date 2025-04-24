#!/bin/bash

well_fov=$1
use_GPU=$2


module load miniforge
conda init bash
conda activate GFF_featurization

cd ../scripts/ || exit

# start the timer
start_timestamp=$(date +%s)
if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"
    python intensity_gpu.py --well_fov $well_fov
else
    echo "Running CPU version"
    python intensity.py --well_fov $well_fov
fi

end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

conda deactivate
