#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=check_job_status-%j.out

job_id_file="job_ids.txt"

mapfile -t job_ids < "$job_id_file"

for job_id_string in "${job_ids[@]}"; do
    job_id=$(echo "$job_id_string" | awk '{print $1}')
    exit_code=$(sacct --jobs="$job_id" --format=exitcode | awk 'NR>2' | head -n 1 | cut -d: -f1)

    # show which jobs do not have an exit code of 0
    if [ "$exit_code" -ne 0 ]; then
        echo "Job $job_id failed with exit code $exit_code"
    fi
done

echo "Checked all jobs for completion"
