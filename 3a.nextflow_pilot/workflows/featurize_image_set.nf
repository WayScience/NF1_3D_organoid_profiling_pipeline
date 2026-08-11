#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

process FEATURIZE_NONGRANULARITY {
  label 'nongranularity_cpu'

  input:
  path manifest_file
  val compartment

  output:
  val compartment

  script:
  def slug = compartment.toLowerCase()
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  """
  set -euo pipefail
  outdir="${params.outdir}/compartments/${slug}/nongranularity"
  mkdir -p "\${outdir}"
  export UV_PROJECT_ENVIRONMENT="${uv_env}"
  export UV_LINK_MODE=copy

  if [[ -n "\${UV_PROJECT_ENVIRONMENT:-}" && -x "\${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    PYTHON_RUNNER=("\${UV_PROJECT_ENVIRONMENT}/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.pilot_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o "\${outdir}/resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.pilot_root}/scripts/run_zedprofiler_image_set.py" \\
    --manifest "${manifest_file}" \\
    --outdir "\${outdir}" \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.pilot_root}/.." \\
    --compartment "${compartment}" \\
    --feature-mode nongranularity \\
    --skip-warehouse
  """
}

process FEATURIZE_GRANULARITY {
  label 'granularity_cpu'

  input:
  path manifest_file
  tuple val(compartment), val(image_channel)

  output:
  tuple val(compartment), val(image_channel)

  script:
  def slug = compartment.toLowerCase()
  def channel_slug = image_channel.toLowerCase()
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  """
  set -euo pipefail
  outdir="${params.outdir}/compartments/${slug}/granularity/${channel_slug}"
  mkdir -p "\${outdir}"
  export UV_PROJECT_ENVIRONMENT="${uv_env}"
  export UV_LINK_MODE=copy

  if [[ -n "\${UV_PROJECT_ENVIRONMENT:-}" && -x "\${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    PYTHON_RUNNER=("\${UV_PROJECT_ENVIRONMENT}/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.pilot_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o "\${outdir}/resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.pilot_root}/scripts/run_zedprofiler_image_set.py" \\
    --manifest "${manifest_file}" \\
    --outdir "\${outdir}" \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.pilot_root}/.." \\
    --compartment "${compartment}" \\
    --channel "${image_channel}" \\
    --feature-mode granularity \\
    --skip-warehouse
  """
}

process BUILD_WAREHOUSE {
  label 'warehouse_cpu'

  input:
  path manifest_file
  val completed_nongranularity
  val completed_granularity

  script:
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  """
  set -euo pipefail
  mkdir -p "${params.outdir}"
  export UV_PROJECT_ENVIRONMENT="${uv_env}"
  export UV_LINK_MODE=copy
  printf '%s\\n' ${completed_nongranularity.join(' ')} > "${params.outdir}/completed_nongranularity.txt"
  printf '%s\\n' ${completed_granularity.collect { it.join(':') }.join(' ')} > "${params.outdir}/completed_granularity.txt"

  if [[ -n "\${UV_PROJECT_ENVIRONMENT:-}" && -x "\${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    PYTHON_RUNNER=("\${UV_PROJECT_ENVIRONMENT}/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.pilot_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o "${params.outdir}/warehouse_resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.pilot_root}/scripts/build_warehouse_from_compartments.py" \\
    --manifest "${manifest_file}" \\
    --compartment-root "${params.outdir}/compartments" \\
    --outdir "${params.outdir}" \\
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

  manifest_file = file(params.manifest)
  compartments = params.compartments.split(',').collect { it.trim() }.findAll { it }
  channels = params.channels.split(',').collect { it.trim() }.findAll { it }

  compartment_ch = Channel.fromList(compartments)
  granularity_ch = Channel.fromList(compartments).combine(Channel.fromList(channels))

  FEATURIZE_NONGRANULARITY(manifest_file, compartment_ch)
  FEATURIZE_GRANULARITY(manifest_file, granularity_ch)
  BUILD_WAREHOUSE(
    manifest_file,
    FEATURIZE_NONGRANULARITY.out.collect(),
    FEATURIZE_GRANULARITY.out.collect(),
  )
}
