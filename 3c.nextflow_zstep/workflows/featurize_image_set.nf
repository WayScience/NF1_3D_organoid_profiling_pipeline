#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Top-level `def` functions, not top-level statements: newer Nextflow
// parsers (confirmed on 26.04.6) reject executable statements mixed with
// process/workflow declarations at file scope. Every one of these recomputes
// its result fresh from params.* each call -- cheap at any realistic image-set
// count -- so both the workflow {} block and every process's script: block
// can call them directly without relying on cross-block variable capture.
//
// 3c.nextflow_zstep vs 3b.nextflow_production: one image set is one
// (patient, well_fov, mask_group) triple, and each FEATURIZE_IMAGE_SET
// task sizes its Slurm request from the image set's z-step. The sizing
// itself lives in one place -- scripts/plan_image_sets.py's
// Z_STEP_RESOURCES -- whose plan lines carry "mem=... cpus=... time=..."
// fields through the pending channel into the task's dynamic memory/cpus/
// time directives below (the documented dynamic-directive pattern:
// closures over input values, which take precedence over the
// withLabel defaults in conf/base.config). Everything else -- PLAN ->
// FEATURIZE fan-out -> BUILD warehouse, skip-already-done,
// write-once-into-outdir -- is unchanged production behavior.

def parseManifestPaths() {
  return params.image_sets
    ? params.image_sets.split(',').collect { it.trim() }.findAll { it }
    : [params.manifest]
}

def slugFor(manifestPath) {
  return file(manifestPath).baseName.toLowerCase().replaceAll(/[^a-z0-9_]+/, '_')
}

def slugForIndexRow(patient, wellFov, maskGroup) {
  return "${patient}_${wellFov}_${maskGroup}".toLowerCase().replaceAll(/[^a-z0-9_]+/, '_')
}

// Every entry is [slug, patient, wellFov, maskGroup, manifestPath]
// regardless of source -- index rows leave manifestPath null, manifest-path
// entries leave patient/wellFov/maskGroup null. Keeping one shape means
// every other function here doesn't need to know which mode is active.
def imageSets() {
  if (params.image_sets_index) {
    def rows = file(params.image_sets_index).splitCsv(header: true)
    def entries = rows.collect { row ->
      def patient = row.patient.trim()
      def wellFov = row.well_fov.trim()
      def maskGroup = row.mask_group.trim()
      [slugForIndexRow(patient, wellFov, maskGroup), patient, wellFov, maskGroup, null]
    }
    def slugs = entries.collect { it[0] }
    if (slugs.unique(false).size() != slugs.size()) {
      error "Duplicate image-set slugs derived from index rows: ${slugs}"
    }
    return entries
  }
  def paths = parseManifestPaths()
  def entries = paths.collect { path -> [slugFor(path), null, null, null, path] }
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
// --manifest PATH, or --patient/--well-fov/--mask-group/--source-root for
// the index-driven path, which derives the manifest on the fly instead of
// reading a YAML file (see build_manifest.py's resolve_manifest()).
def manifestArgsForSlug(slug) {
  def (_slug, patient, wellFov, maskGroup, manifestPath) = entryForSlug(slug)
  if (manifestPath) {
    return "--manifest \"${file(manifestPath)}\""
  }
  return "--patient \"${patient}\" --well-fov \"${wellFov}\" --mask-group \"${maskGroup}\" --source-root \"${params.source_root}\""
}

process PLAN_IMAGE_SETS {
  // Checks warehouse/ + metadata/ directly for each image set rather than
  // relying on Nextflow's -resume: this workflow never configures a durable
  // workDir across separate `make submit` invocations, so there is nothing
  // for -resume to resume from, and even with one, its cache key is task
  // inputs, not "does a finished, valid output already exist". Runs once
  // per invocation, before any FEATURIZE_IMAGE_SET task is submitted.

  output:
  path 'image_set_plan.tsv'

  script:
  def uv_env = params.uv_project_environment ?: '${PWD}/.venv'
  def plan_args = params.image_sets_index
    ? "--image-sets-index \"${file(params.image_sets_index)}\" --source-root \"${params.source_root}\""
    : imageSets().collect { slug, patient, wellFov, maskGroup, manifestPath ->
        "--manifest \"${file(manifestPath)}\""
      }.join(' \\\n    ')
  """
  set -euo pipefail
  export UV_PROJECT_ENVIRONMENT="${uv_env}"
  export UV_LINK_MODE=copy

  if [[ -n "\${UV_PROJECT_ENVIRONMENT:-}" && -x "\${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    PYTHON_RUNNER=("\${UV_PROJECT_ENVIRONMENT}/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER=(uv run --project "${params.workflow_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  "\${PYTHON_RUNNER[@]}" "${params.workflow_root}/scripts/plan_image_sets.py" \\
    ${plan_args} \\
    --outdir "${params.outdir}" \\
    > image_set_plan.tsv
  """
}

process FEATURIZE_IMAGE_SET {
  label 'zedprofiler_cpu'

  // Dynamic per-task sizing: a 0.1 um image set is ~10x the voxel count of
  // a 1 um one, so each task requests memory/cpus/time from the plan line
  // plan_image_sets.py emitted for its z-step (Z_STEP_RESOURCES is the
  // single source of truth). Directive closures over input values take
  // precedence over the withLabel: zedprofiler_cpu defaults in
  // conf/base.config.
  memory { task_memory }
  cpus { task_cpus }
  time { task_time }

  input:
  tuple val(image_set_slug), val(task_memory), val(task_cpus), val(task_time)

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
    PYTHON_RUNNER=(uv run --project "${params.workflow_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  # run_with_time: GNU time -v writes peak-RSS/walltime to the given file
  # for post-run sacct cross-checks; on hosts without GNU time (macOS
  # /usr/bin/time rejects -v and --version alike) plain-exec so the local
  # profile still works, just without the usage file.
  run_with_time() {
    local out="\$1"; shift
    if [[ -x /usr/bin/time ]] && /usr/bin/time --version >/dev/null 2>&1; then
      /usr/bin/time -v -o "\$out" "\$@"
    else
      "\$@"
    fi
  }

  run_with_time "resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.workflow_root}/scripts/run_zedprofiler_image_set.py" \\
    ${manifest_args} \\
    --outdir "${params.outdir}" \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.workflow_root}/.."
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
    : imageSets().collect { slug, patient, wellFov, maskGroup, manifestPath ->
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
    PYTHON_RUNNER=(uv run --project "${params.workflow_root}/environments" python)
  else
    PYTHON_RUNNER=(python3)
  fi

  # run_with_time: see FEATURIZE_IMAGE_SET -- GNU time -v when available,
  # plain execution otherwise (macOS local profile).
  run_with_time() {
    local out="\$1"; shift
    if [[ -x /usr/bin/time ]] && /usr/bin/time --version >/dev/null 2>&1; then
      /usr/bin/time -v -o "\$out" "\$@"
    else
      "\$@"
    fi
  }

  run_with_time "${params.outdir}/warehouse_resource_usage.txt" "\${PYTHON_RUNNER[@]}" \\
    "${params.workflow_root}/scripts/build_warehouse_from_compartments.py" \\
    ${manifest_args} \\
    --outdir "${params.outdir}" \\
    --run-id "${params.run_id ?: 'manual'}" \\
    --repo-root "${params.workflow_root}/.."
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
  if (!params.workflow_root) {
    error "params.workflow_root is required"
  }

  PLAN_IMAGE_SETS()

  // Plan lines are "slug\tstatus\tmem=...\tcpus=...\ttime=..."; each pending
  // row becomes a [slug, mem, cpus, time] tuple the FEATURIZE task reads
  // its per-z-step Slurm sizing from (see FEATURIZE_IMAGE_SET's directives).
  pending_entries = PLAN_IMAGE_SETS.out
    .splitCsv(sep: '\t')
    .filter { row -> row[1] == 'pending' }
    .map { row ->
      def fields = [:]
      row[2..-1].each { field ->
        def (key, value) = field.split('=', 2)
        fields[key] = value
      }
      tuple(row[0], fields['mem'], fields['cpus'] as int, fields['time'])
    }

  FEATURIZE_IMAGE_SET(pending_entries)

  // ifEmpty() matters here, not just style: if every image set was already
  // done and PLAN_IMAGE_SETS skipped all of them, FEATURIZE_IMAGE_SET never
  // runs a single task, and .collect() on that untouched output channel
  // never fires -- BUILD_WAREHOUSE would silently never run at all (caught
  // by a local test of exactly this all-skipped case, not assumed).
  BUILD_WAREHOUSE(FEATURIZE_IMAGE_SET.out.collect().ifEmpty(['none_pending']))
}
