SHELL := /usr/bin/env bash
export PYENV_VERSION ?= system

RUN_ID ?=
MANIFEST ?= manifest/smoke_test_image_set.yaml
MANIFEST_OUTPUT ?= manifest/generated_manifest.yaml
IMAGE_SETS ?=
IMAGE_SETS_INDEX ?= manifest/image_sets_index.csv
INDEX_OUTPUT ?= manifest/image_sets_index.csv
SOURCE_ROOT ?= /pl/active/koala/nf1-3d-zstep-workflow-db
RESULTS_ROOT ?= /pl/active/koala/nf1-3d-zstep-workflow-db/results
PATIENT ?=
WELL_FOV ?=
MASK_GROUP ?=
PROFILE ?= local
ACCOUNT ?=
PARTITION ?= acpu
QOS ?= cpu-normal
SUBMIT_HOST ?=
UV_PROJECT_ENVIRONMENT ?=
FEATURE_CPUS ?= 2
FEATURE_MEMORY ?= 32 GB
FEATURE_TIME ?= 2h
COORDINATOR_TIME ?= 1-00:00:00

MANIFEST_ARGS :=
ifneq ($(strip $(SOURCE_ROOT)),)
MANIFEST_ARGS += --source-root "$(SOURCE_ROOT)"
endif
ifneq ($(strip $(PATIENT)),)
MANIFEST_ARGS += --patient "$(PATIENT)"
endif
ifneq ($(strip $(WELL_FOV)),)
MANIFEST_ARGS += --well-fov "$(WELL_FOV)"
endif
ifneq ($(strip $(MASK_GROUP)),)
MANIFEST_ARGS += --mask-group "$(MASK_GROUP)"
endif

RUN_ARGS :=
ifneq ($(strip $(RUN_ID)),)
RUN_ARGS += --run-id "$(RUN_ID)"
endif
ifneq ($(strip $(UV_PROJECT_ENVIRONMENT)),)
RUN_ARGS += --uv-project-environment "$(UV_PROJECT_ENVIRONMENT)"
endif
ifneq ($(strip $(IMAGE_SETS_INDEX)),)
RUN_ARGS += --image-sets-index "$(IMAGE_SETS_INDEX)" --source-root "$(SOURCE_ROOT)"
else ifneq ($(strip $(IMAGE_SETS)),)
RUN_ARGS += --image-sets "$(IMAGE_SETS)"
else
RUN_ARGS += --manifest "$(MANIFEST)"
endif
RUN_ARGS += --results-root "$(RESULTS_ROOT)"
# NOTE: FEATURIZE_IMAGE_SET memory/cpus/time are sized per image set from
# its z-step by plan_image_sets.py's Z_STEP_RESOURCES table -- the .nf
# workflow reads each task's request from the plan line. These FEATURE_*
# variables remain as fallbacks (e.g. --manifest runs without a plan) and
# size the coordinator's own defaults.
RUN_ARGS += --feature-cpus "$(FEATURE_CPUS)"
RUN_ARGS += --feature-memory "$(FEATURE_MEMORY)"
RUN_ARGS += --feature-time "$(FEATURE_TIME)"

.PHONY: help check doctor preflight clear-output clear-output-dry-run build-manifest validate-manifest build-image-sets-index run submit submit-dry-run smoke clean

help:
	@bash bin/nf1-nextflow-zstep --help

check:
	@bash bin/nf1-nextflow-zstep check

doctor:
	@bash bin/nf1-nextflow-zstep doctor

preflight:
	@bash bin/nf1-nextflow-zstep preflight $(if $(RUN_ID),--run-id "$(RUN_ID)") --results-root "$(RESULTS_ROOT)" --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)"

clear-output:
	@bash bin/nf1-nextflow-zstep clear-output $(if $(RUN_ID),--run-id "$(RUN_ID)") --results-root "$(RESULTS_ROOT)"

clear-output-dry-run:
	@bash bin/nf1-nextflow-zstep clear-output $(if $(RUN_ID),--run-id "$(RUN_ID)") --results-root "$(RESULTS_ROOT)" --dry-run

build-manifest:
	@python3 scripts/build_manifest.py $(MANIFEST_ARGS) --output "$(MANIFEST_OUTPUT)"

validate-manifest:
	@python3 scripts/build_manifest.py --validate-only --output "$(MANIFEST)"

build-image-sets-index:
	@python3 scripts/build_image_sets_index.py --source-root "$(SOURCE_ROOT)" --output "$(INDEX_OUTPUT)"

run:
	@NF1_ZSTEP_ACCOUNT="$(ACCOUNT)" NF1_ZSTEP_PARTITION="$(PARTITION)" NF1_ZSTEP_QOS="$(QOS)" \
	  bash bin/nf1-nextflow-zstep run --profile "$(PROFILE)" $(RUN_ARGS) --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)"

submit:
	@bash bin/nf1-nextflow-zstep submit $(RUN_ARGS) --coordinator-time "$(COORDINATOR_TIME)" --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)" $(if $(SUBMIT_HOST),--submit-host "$(SUBMIT_HOST)")

submit-dry-run:
	@bash bin/nf1-nextflow-zstep submit $(RUN_ARGS) --coordinator-time "$(COORDINATOR_TIME)" --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)" $(if $(SUBMIT_HOST),--submit-host "$(SUBMIT_HOST)") --dry-run

smoke:
	@bash bin/nf1-nextflow-zstep smoke $(if $(RUN_ID),--run-id "$(RUN_ID)")

clean:
	@printf 'Generated z-step artifacts default to $(RESULTS_ROOT).\n'
