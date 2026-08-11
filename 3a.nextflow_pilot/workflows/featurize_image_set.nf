#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

process FEATURIZE_IMAGE_SET {
  label 'zedprofiler_cpu'
  publishDir params.outdir, mode: 'copy', overwrite: true

  input:
  path manifest_file

  output:
  path '*_profiles.parquet'
  path 'warehouse'
  path 'run_record.json'
  path 'warehouse_manifest.json'
  path 'validation.json'
  path 'alignment_validation.json'
  path 'resource_usage.txt'

  script:
  """
  set -euo pipefail

  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.pilot_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o resource_usage.txt "\${PYTHON_RUNNER[@]}" \\
    "${params.pilot_root}/scripts/run_zedprofiler_image_set.py" \\
    --manifest "${manifest_file}" \\
    --outdir . \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.pilot_root}/.."
  """
}

workflow {
  if (!params.manifest) {
    error "params.manifest is required"
  }
  if (!params.outdir) {
    error "params.outdir is required"
  }
  if (!params.pilot_root) {
    error "params.pilot_root is required"
  }
  FEATURIZE_IMAGE_SET(file(params.manifest))
}
