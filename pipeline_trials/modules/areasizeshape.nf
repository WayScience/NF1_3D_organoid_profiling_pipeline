process AREASIZESHAPE_CPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "areasizeshape_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
    bash run_area_shape_child.sh ${well_fov} ${featurize_with_gpu} ${patient}
    cd ${baseDir}/ || exit 1
    """
}

process AREASIZESHAPE_GPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'

    tag { "areasizeshape_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
    bash run_area_shape_child.sh ${patient} ${well_fov} TRUE ${featurize_with_gpu}
    cd ${baseDir}/ || exit 1
    """
}
