#!/usr/bin/env nextflow
// This is a nexflow script for running the GFF featurization pipeline
// It is designed to run on a SLURM cluster or locally
// Note that all code written here is written in nextflow DSL2
// DSL2 = Domain Specific Language 2
// For this pipeline to run it is assumed that segmentation has already been run

nextflow.enable.dsl = 2

params.fov_file = 'patient_well_fov.tsv'
params.featurize_with_gpu = false

include { AREASIZESHAPE_CPU; AREASIZESHAPE_GPU } from './modules/areasizeshape.nf'
include { COLOCALIZATION_CPU; COLOCALIZATION_GPU } from './modules/colocalization.nf'
include { GRANULARITY_CPU; GRANULARITY_GPU } from './modules/granularity.nf'
include { INTENSITY_CPU; INTENSITY_GPU } from './modules/intensity.nf'
include { NEIGHBORS_CPU } from './modules/neighbors.nf'
include { TEXTURE_CPU } from './modules/texture.nf'


workflow {
    // Common channel from input file
    def fov_ch = Channel
        .fromPath(params.fov_file)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(row.patient, row.well_fov, params.featurize_with_gpu ?: false) // Ensure featurize_with_gpu is not null
        }

    // Run segmentation (shared between all)
    def segmented_ch = fov_ch.map { patient, well_fov, _ -> tuple(patient, well_fov) }

    // Re-attach featurize_with_gpu flag
    def full_ch = segmented_ch.map { patient, well_fov ->
        tuple(patient, well_fov, params.featurize_with_gpu)
    }

    // if featurize_with_gpu is false, run CPU branches
    def cpu_ch = full_ch.filter { patient, well_fov, featurize_with_gpu -> !featurize_with_gpu }
    // if featurize_with_gpu is true, run GPU branches
    def gpu_ch = full_ch.filter { patient, well_fov, featurize_with_gpu -> featurize_with_gpu }
    // Run GPU branches
    gpu_ch | AREASIZESHAPE_GPU
    gpu_ch | COLOCALIZATION_GPU
    gpu_ch | GRANULARITY_GPU
    gpu_ch | INTENSITY_GPU

    // Run CPU branches
    cpu_ch | AREASIZESHAPE_CPU
    cpu_ch | COLOCALIZATION_CPU
    cpu_ch | GRANULARITY_CPU
    cpu_ch | INTENSITY_CPU

    // Run neighbors on CPU
    segmented_ch.map { patient, well_fov -> tuple(patient, well_fov, false) } | NEIGHBORS_CPU
    // Always run texture on CPU
    segmented_ch.map { patient, well_fov -> tuple(patient, well_fov, false) } | TEXTURE_CPU
}
