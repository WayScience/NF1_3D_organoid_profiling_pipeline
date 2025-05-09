process COLOCALIZATION_CPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'

    tag { "colocalization_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    bash run_colocalization_child.sh ${patient} FALSE ${well_fov}
    cd ${baseDir}/ || exit 1
    """
}

process COLOCALIZATION_GPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'

    tag { "colocalization_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    bash run_colocalization_child.sh ${patient} TRUE ${well_fov}
    cd ${baseDir}/ || exit 1
    """
}
