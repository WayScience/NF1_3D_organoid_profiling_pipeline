#!/bin/bash

patient=$1

cd scripts/ || exit


number_of_jobs=$(squeue -u $USER | wc -l)
while [ $number_of_jobs -gt 990 ]; do
    sleep 1s
    number_of_jobs=$(squeue -u $USER | wc -l)
done
echo "Cleaning up segmentation files for patient: $patient"
python 9.clean_up_segmentation.py --patient "$patient"



cd ../ || exit

conda deactivate
echo "Segmentation cleanup completed"
