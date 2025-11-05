#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=setup_envs-%j.out

# Load your module for conda/mamba if needed
module load miniforge

# Run Makefile
make --always-make
