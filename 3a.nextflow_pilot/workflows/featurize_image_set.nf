#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Top-level `def` functions, not top-level statements: newer Nextflow
// parsers (confirmed on 26.04.6) reject executable statements mixed with
// process/workflow declarations at file scope. Every one of these recomputes
// its result fresh from params.* each call -- cheap at any realistic image-set
// count -- so both the workflow {} block and every process's script: block
// can call them directly without relying on cross-block variable capture.

def parseManifestPaths() {
  return params.image_sets
    ? params.image_sets.split(',').collect { it.trim() }.findAll { it }
    : [params.manifest]
}

def slugFor(manifestPath) {
  return file(manifestPath).baseName.toLowerCase().replaceAll(/[^a-z0-9_]+/, '_')
}

def imageSets() {
  def paths = parseManifestPaths()
  def slugs = paths.collect { slugFor(it) }
  if (slugs.unique(false).size() != slugs.size()) {
    error "Duplicate image-set slugs derived from manifest filenames: ${slugs}"
  }
  return [slugs, paths].transpose()
}

def manifestPathForSlug(slug) {
  def match = imageSets().find { it[0] == slug }
  if (!match) {
    error "Unknown image-set slug: ${slug}"
  }
  return file(match[1])
}

process FEATURIZE_IMAGE_SET {
  label 'zedprofiler_cpu'

  input:
  val image_set_slug

  output:
  val image_set_slug

  script:
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  def manifest_path = manifestPathForSlug(image_set_slug)
  """
  set -euo pipefail
  export UV_PROJECT_ENVIRONMENT="${uv_env}"
  export UV_LINK_MODE=copy

  if [[ -n "\${UV_PROJECT_ENVIRONMENT:-}" && -x "\${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    PYTHON_RUNNER=("\${UV_PROJECT_ENVIRONMENT}/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.pilot_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o "resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.pilot_root}/scripts/run_zedprofiler_image_set.py" \\
    --manifest "${manifest_path}" \\
    --outdir "${params.outdir}" \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.pilot_root}/.."
  """
}

process BUILD_WAREHOUSE {
  label 'warehouse_cpu'

  input:
  val completed_image_sets

  script:
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  def manifest_args = imageSets().collect { slug, manifest_path ->
    "--manifest \"${file(manifest_path)}\""
  }.join(' \\\n    ')
  """
  set -euo pipefail
  mkdir -p "${params.outdir}"
  export UV_PROJECT_ENVIRONMENT="${uv_env}"
  export UV_LINK_MODE=copy
  printf '%s\\n' ${completed_image_sets.collect { it.toString() }.join(' ')} > "${params.outdir}/completed_image_sets.txt"

  if [[ -n "\${UV_PROJECT_ENVIRONMENT:-}" && -x "\${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    PYTHON_RUNNER=("\${UV_PROJECT_ENVIRONMENT}/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.pilot_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  /usr/bin/time -v -o "${params.outdir}/warehouse_resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.pilot_root}/scripts/build_warehouse_from_compartments.py" \\
    ${manifest_args} \\
    --outdir "${params.outdir}" \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.pilot_root}/.."
  """
}

workflow {
  if (!params.manifest && !params.image_sets) {
    error "params.manifest or params.image_sets is required"
  }
  if (!params.outdir) {
    error "params.outdir is required"
  }
  if (!params.pilot_root) {
    error "params.pilot_root is required"
  }

  image_sets = imageSets()
  image_set_slugs = image_sets.collect { slug, path -> slug }

  FEATURIZE_IMAGE_SET(Channel.fromList(image_set_slugs))

  BUILD_WAREHOUSE(FEATURIZE_IMAGE_SET.out.collect())
}
