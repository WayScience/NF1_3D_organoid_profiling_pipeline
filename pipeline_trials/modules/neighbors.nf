process NEIGHBORS_CPU {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'

    tag { "neighbors_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running GPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    bash run_neighbors_child.sh ${well_fov} FALSE ${patient}
    cd ${baseDir}/ || exit 1
    """
}
