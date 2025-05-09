#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=3-00:00:00
#SBATCH --output=output_for_nf_featurization-%j.out

module load nextflow

HPC_RUN=False
FEATS_ONLY=True


if [ "$HPC_RUN" = "True" ]; then
    if [ "$FEATS_ONLY" = "True" ]; then
        nextflow \
            featurization_only.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --featurize_with_gpu "/projects/mlippincott@xsede.org/software/anaconda/envs/" \
            -c ./configs/nextflow_hpc.config
    else
        nextflow \
            segmentation_through_featurization.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --featurize_with_gpu "/projects/mlippincott@xsede.org/software/anaconda/envs/" \
            -c ./configs/nextflow_hpc.config
    fi
else
    if [ "$FEATS_ONLY" = "True" ]; then
        nextflow \
            featurization_only.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --featurize_with_gpu "/home/lippincm/miniforge3/envs/" \
            -c ./configs/nextflow_local.config
    else
        nextflow \
            segmentation_through_featurization.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --featurize_with_gpu "/home/lippincm/miniforge3/envs/" \
            -c ./configs/nextflow_local.config
    fi
fi
