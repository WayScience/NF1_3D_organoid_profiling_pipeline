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

########################################################################################
## Clean up the segmentation process and prepare for feature extraction
########################################################################################
sbatch 3b.process_segmentation_child.sh

echo "Array complete"

# end this job once reaching this point
exit 0
