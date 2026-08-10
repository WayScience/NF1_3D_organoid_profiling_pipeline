#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

def outdir = params.outdir ?: "results/${params.run_id ?: 'manual'}"

if (!params.manifest) {
  error "params.manifest is required"
}

process FEATURIZE_IMAGE_SET {
  label 'zedprofiler_cpu'
  publishDir outdir, mode: 'copy', overwrite: true

  input:
  path manifest_file

  output:
  path 'nuclei_profiles.parquet'
  path 'run_record.json'
  path 'warehouse_manifest.json'
  path 'validation.json'
  path 'resource_usage.txt'

  script:
  """
  set -euo pipefail

  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "$baseDir/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o resource_usage.txt "\${PYTHON_RUNNER[@]}" \\
    "$baseDir/scripts/run_zedprofiler_image_set.py" \\
    --manifest "${manifest_file}" \\
    --outdir . \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "$baseDir/.."
  """
}

workflow {
  FEATURIZE_IMAGE_SET(file(params.manifest))
}

