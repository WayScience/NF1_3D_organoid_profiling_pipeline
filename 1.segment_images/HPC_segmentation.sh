#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=7-00:00:00
#SBATCH --output=main_job_submit_output-%j.out

cd slurm_scripts || exit

job1=$(sbatch 0.inital_segmentation_HPC.sh | awk '{print $4}')

job2=$(sbatch --dependency=afterok:$job1 1a.process_segmentation_parent.sh | awk '{print $4}')

job3=$(sbatch --dependency=afterok:$job2 2a.process_segmentation_parent.sh | awk '{print $4}')

job4=$(sbatch --dependency=afterok:$job3 3a.process_segmentation_parent.sh | awk '{print $4}')

job5=$(sbatch --dependency=afterok:$job4 4.check_job_status.sh | awk '{print $4}')

echo "All jobs submitted"

exit 0
