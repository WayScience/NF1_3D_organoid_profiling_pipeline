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

def slugForIndexRow(patient, wellFov) {
  return "${patient}_${wellFov}".toLowerCase().replaceAll(/[^a-z0-9_]+/, '_')
}

// Every entry is [slug, patient, wellFov, manifestPath] regardless of
// source -- index rows leave manifestPath null, manifest-path entries
// leave patient/wellFov null. Keeping one shape means every other
// function here doesn't need to know which mode is active.
def imageSets() {
  if (params.image_sets_index) {
    def rows = file(params.image_sets_index).splitCsv(header: true)
    def entries = rows.collect { row ->
      def patient = row.patient.trim()
      def wellFov = row.well_fov.trim()
      [slugForIndexRow(patient, wellFov), patient, wellFov, null]
    }
    def slugs = entries.collect { it[0] }
    if (slugs.unique(false).size() != slugs.size()) {
      error "Duplicate image-set slugs derived from index rows: ${slugs}"
    }
    return entries
  }
  def paths = parseManifestPaths()
  def entries = paths.collect { path -> [slugFor(path), null, null, path] }
  def slugs = entries.collect { it[0] }
  if (slugs.unique(false).size() != slugs.size()) {
    error "Duplicate image-set slugs derived from manifest filenames: ${slugs}"
  }
  return entries
}

def entryForSlug(slug) {
  def match = imageSets().find { it[0] == slug }
  if (!match) {
    error "Unknown image-set slug: ${slug}"
  }
  return match
}

// CLI args identifying one image set to a Python script -- either
// --manifest PATH, or --patient/--well-fov/--source-root for the
// index-driven path, which derives the manifest on the fly instead of
// reading a YAML file (see build_manifest.py's resolve_manifest()).
def manifestArgsForSlug(slug) {
  def (_slug, patient, wellFov, manifestPath) = entryForSlug(slug)
  if (manifestPath) {
    return "--manifest \"${file(manifestPath)}\""
  }
  return "--patient \"${patient}\" --well-fov \"${wellFov}\" --source-root \"${params.source_root}\""
}

process FEATURIZE_IMAGE_SET {
  label 'zedprofiler_cpu'

  input:
  val image_set_slug

  output:
  val image_set_slug

  script:
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  def manifest_args = manifestArgsForSlug(image_set_slug)
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
    ${manifest_args} \\
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
  def manifest_args = params.image_sets_index
    ? "--image-sets-index \"${file(params.image_sets_index)}\" --source-root \"${params.source_root}\""
    : imageSets().collect { slug, patient, wellFov, manifestPath ->
        "--manifest \"${file(manifestPath)}\""
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
  if (!params.manifest && !params.image_sets && !params.image_sets_index) {
    error "params.manifest, params.image_sets, or params.image_sets_index is required"
  }
  if (params.image_sets_index && !params.source_root) {
    error "params.source_root is required when using params.image_sets_index"
  }
  if (!params.outdir) {
    error "params.outdir is required"
  }
  if (!params.pilot_root) {
    error "params.pilot_root is required"
  }

  image_sets = imageSets()
  image_set_slugs = image_sets.collect { slug, patient, wellFov, manifestPath -> slug }

  FEATURIZE_IMAGE_SET(Channel.fromList(image_set_slugs))

  BUILD_WAREHOUSE(FEATURIZE_IMAGE_SET.out.collect())
}
