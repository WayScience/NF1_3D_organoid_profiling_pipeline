#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=parent-output-%j.out


module load anaconda

conda activate GFF_segmentation


# get all input directories in specified directory
z_stack_dir="../../data/z-stack_images/"
# z_stack_dir="../../data/test_dir/"


# Use mapfile to read the output of ls -d into an array
mapfile -t input_dirs < <(ls -d "$z_stack_dir"*)

# Print each path to ensure they are separate elements
for dir in "${input_dirs[@]}"; do
    echo "Directory: $dir"
done

compartments=( "nuclei" "cell" )

touch job_ids.txt
jobs_submitted_counter=0
########################################################################################
## Finish the segmentation process for each compartment
########################################################################################
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
        job_id=$(sbatch 1a.process_segmentation_child.sh "$dir" "$compartment")
        # append the job id to the file
        job_id=$(echo $job_id | awk '{print $4}')
        let jobs_submitted_counter++
	done
done

# wait for all jobs to finish before proceeding
while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 1s
done

########################################################################################
## Finish the segmentation process for the cytoplasm
## This ensures that the cell and nuclei masks are created before the cytoplasm masks
########################################################################################
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
    job_id=$(sbatch 1b.process_segmentation_child.sh "$dir")
    # append the job id to the file
    job_id=$(echo $job_id | awk '{print $4}')
    let jobs_submitted_counter++
done

# wait for all jobs to finish before proceeding
while [ $(squeue -u $USER | wc -l) -gt 2 ]; do
    sleep 1s
done

########################################################################################
## Clean up the segmentation process and prepare for feature extraction
########################################################################################
job_id=$(sbatch 1c.process_segmentation_child.sh | awk '{print $4}')

sbatch --dependency=afterok:"$job_id" 2.check_job_status.sh

cd ../ || exit

echo "$jobs_submitted_counter"

echo "Array complete"

# end this job once reaching this point
exit 0
