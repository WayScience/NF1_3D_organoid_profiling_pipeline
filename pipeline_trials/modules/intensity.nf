
process INTENSITY_CPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'

    tag { "intensity_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running GPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    bash run_intensity_child.sh ${well_fov} FALSE ${patient}
    cd ${baseDir}/ || exit 1
    """
}

process INTENSITY_GPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'

    tag { "intensity_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running GPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    bash run_intensity_child.sh ${well_fov} TRUE ${patient}
    cd ${baseDir}/ || exit 1
    """
}
