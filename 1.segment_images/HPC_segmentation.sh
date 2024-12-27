#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=main_job_submit_output-%j.out

cd slurm_scripts || exit

job1=$(sbatch inital_semgentation_HPC.sh | awk '{print $4}')

job2=$(sbatch --dependency=afterok:$job1 process_semgentation_parent_pt1.sh | awk '{print $4}')

job3=$(sbatch --dependency=afterok:$job2 process_semgentation_parent_pt2.sh | awk '{print $4}')

exit 0
