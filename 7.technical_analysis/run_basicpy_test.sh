#!/bin/bash

# --------------------------
# CONFIG
# --------------------------
NOTEBOOK="test_basicpy.ipynb"           # your notebook filename
PATIENT_ID="NF0037_T1_CQ1"                     # patient to process
NBCONVERT_DIR="nbconverted"                # folder to save converted scripts
ENV_NAME="NF1_3D_basicpy_env"             # mamba/conda environment to activate

# --------------------------
# 0️⃣ Activate environment
# --------------------------
# Make sure 'mamba' or 'conda' is in your PATH
conda init bash
conda activate "$ENV_NAME"

# --------------------------
# 1️⃣ Convert notebook to Python script
# --------------------------
mkdir -p "$NBCONVERT_DIR"
jupyter nbconvert --to script "notebooks/$NOTEBOOK" --output-dir "$NBCONVERT_DIR"

# Get the path to the converted script
SCRIPT_NAME=$(basename "$NOTEBOOK" .ipynb).py
SCRIPT_PATH="$NBCONVERT_DIR/$SCRIPT_NAME"

# --------------------------
# 2️⃣ Run the converted script with patient ID
# --------------------------
python "$SCRIPT_PATH" "$PATIENT_ID"
