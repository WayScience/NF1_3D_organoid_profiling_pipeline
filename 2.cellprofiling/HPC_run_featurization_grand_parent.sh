#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=featurization_sc_grand_parent-%j.out


module load anaconda
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

USE_GPU="FALSE"

parent_dir="../../data/NF0014/cellprofiler"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)

# loop through each dir and submit a job
for dir in $dirs; do
    well_fov=$(basename $dir)
    echo $well_fov
    # check that the number of jobs is less than 990
    # prior to submitting a job
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch HPC_run_featurization_parent.sh "$well_fov" $USE_GPU
    sleep 10s # avoid stacking the parent jobs rather - let us stack the child jobs
done


cd ../ || exit

conda deactivate

echo "Featurization done"
