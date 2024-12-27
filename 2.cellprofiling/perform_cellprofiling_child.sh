#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=cp_child-%j.out

# each node here has 3.75GB of memory.
# in testing, one image set peaked at 11.9 GB of memory usage.
# I will request 4 cores to be safe.
# Where 4 cores is 15GB of memory.

# In testing one iamge set peaked at 00:16:19.30 (16 minutes) of runtime.
# I will request 30 hours to be safe.
# Each image set will be processed in parallel (embarrassingly parallel).

# activate cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_cellprofiler

dir=$1

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python run_cellprofiler_analysis.py --input_dir $dir

cd .. || exit

# deactivate cellprofiler environment
conda deactivate

echo "Cellprofiler analysis done"
