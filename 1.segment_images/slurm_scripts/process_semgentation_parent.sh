#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=parent-output-%j.out


module load anaconda

conda activate GFF_segmentation


# get all input directories in specified directory
z_stack_dir="../../data/z-stack_images/"


# Use mapfile to read the output of ls -d into an array
mapfile -t input_dirs < <(ls -d "$z_stack_dir"*)

# Print each path to ensure they are separate elements
for dir in "${input_dirs[@]}"; do
    echo "Directory: $dir"
done

echo $input_dir

compartments=( "nuclei" "cell" )

touch job_ids.txt
jobs_submitted_counter=0
for compartment in "${compartments[@]}"; do
    for dir in "${input_dirs[@]}"; do
        dir=${dir%*/}
	# get the number of jobs for the user
        number_of_jobs=$(squeue -u $USER | wc -l)
        while [ $number_of_jobs -gt 990 ]; do
            sleep 1s
            number_of_jobs=$(squeue -u $USER | wc -l)
        done
	echo " '$job_id' '$compartment' '$dir' "
        echo " '$job_id' '$compartment' '$dir' " >> job_ids.txt
        job_id=$(sbatch process_semgentation_child_pt1.sh "$dir" "$compartment")
        # append the job id to the file
        job_id=$(echo $job_id | awk '{print $4}')
        let jobs_submitted_counter++
	done
done

# wait for all jobs to finish before proceeding
while [ $(squeue -u $USER | wc -l) -gt 1 ]; do
    sleep 1s
done

for dir in "${input_dirs[@]}"; do
    dir=${dir%*/}
# get the number of jobs for the user
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    echo " '$job_id' '$dir' "
    echo " '$job_id' '$dir' " >> job_ids.txt
    job_id=$(sbatch process_semgentation_child_pt2.sh "$dir")
    # append the job id to the file
    job_id=$(echo $job_id | awk '{print $4}')
    let jobs_submitted_counter++
done

# wait for all jobs to finish before proceeding
while [ $(squeue -u $USER | wc -l) -gt 1 ]; do
    sleep 1s
done

cd ../slurm_scripts/ || exit

sbatch process_semgentation_child_pt3.sh

cd ../ || exit

echo "$jobs_submitted_counter"

echo "Array complete"

# end this job once reaching this point
exit 0
