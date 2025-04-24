#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_featurization

cd ../scripts/ || exit
well_fov=$1
# start the timer
start_timestamp=$(date +%s)
python neighbors.py --well_fov $well_fov
end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

conda deactivate
