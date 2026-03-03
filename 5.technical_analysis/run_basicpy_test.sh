#!/usr/bin/env bash
# --------------------------
# CONFIG
# --------------------------
NOTEBOOK="test_basicpy.ipynb"      # your notebook filename
NBCONVERT_DIR="scripts"        # folder to save converted scripts
ENV_NAME="NF1_3D_basicpy_env"      # mamba/conda environment to activate
patient_id_file_path="../data/patient_IDs.txt"  # path to text file containing patient IDs (one per line)
readarray -t patients < "$patient_id_file_path"  # read patient IDs into an array
# --------------------------
# Resolve paths relative to script
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_PATH="$SCRIPT_DIR/notebooks/$NOTEBOOK"
OUTPUT_DIR="$SCRIPT_DIR/$NBCONVERT_DIR"

# --------------------------
# 0️⃣ Activate environment
# --------------------------
conda init bash
conda activate "$ENV_NAME"

# --------------------------
# 1️⃣ Convert notebook to Python script
# --------------------------
mkdir -p "$OUTPUT_DIR"
jupyter nbconvert --to script "$NOTEBOOK_PATH" --output-dir "$OUTPUT_DIR"

# Get the path to the converted script
SCRIPT_NAME="$(basename "$NOTEBOOK" .ipynb).py"
SCRIPT_PATH="$OUTPUT_DIR/$SCRIPT_NAME"

# --------------------------
# 2️⃣ Run the converted script with patient ID
# --------------------------
for patient_id in "${patients[@]}"; do
    echo "Running script for patient: $patient_id"
    python "$SCRIPT_PATH" "$patient_id"
done

