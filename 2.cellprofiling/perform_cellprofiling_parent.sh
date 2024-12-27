#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=../cp-%j.out

# activate cellprofiler environment
module load anaconda
conda init bash
conda activate cellprofiler_timelapse_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# cellprofiler data directory
mapfile -t data_dirs < <(ls -d ../../data/cellprofiler/*)
cd .. || exit

jobs_submitted_counter=0
for dir in "${data_dirs[@]}"; do
        dir=${dir%*/}
        # get the number of jobs for the user
        number_of_jobs=$(squeue -u $USER | wc -l)
        while [ $number_of_jobs -gt 990 ]; do
            sleep 1s
            number_of_jobs=$(squeue -u $USER | wc -l)
        done
        echo " '$job_id' '$dir' "
        echo " '$job_id' '$dir' " >> job_ids.txt
        job_id=$(sbatch perform_cellprofiling_child.sh "$dir")
        # append the job id to the file
        job_id=$(echo $job_id | awk '{print $4}')
        let jobs_submitted_counter++
done

# deactivate cellprofiler environment
conda deactivate

echo "Cellprofiler analysis done"
