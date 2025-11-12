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
module load make

INSTALL_OR_UPDATE="install-all"
# Run Makefile
if [ "$INSTALL_OR_UPDATE" == "install-all" ]; then
    make --install-all
elif [ "$INSTALL_OR_UPDATE" == "update-all" ]; then
    make --update-all
else
    echo "Invalid option for INSTALL_OR_UPDATE. Use 'install-all' or 'update-all'."
    exit 1
fi
