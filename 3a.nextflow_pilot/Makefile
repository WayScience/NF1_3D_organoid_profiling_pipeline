SHELL := /usr/bin/env bash

RUN_ID ?=
MANIFEST ?= manifest/smoke_test_image_set.yaml
SOURCE_ROOT ?=
PATIENT ?= NF0014_T1
WELL_FOV ?=
PROFILE ?= local
ACCOUNT ?=
PARTITION ?= acpu
QOS ?= cpu-normal
SUBMIT_HOST ?=
FEATURE_CPUS ?= 2
FEATURE_MEMORY ?= 32 GB
FEATURE_TIME ?= 2h

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

RUN_ARGS :=
ifneq ($(strip $(RUN_ID)),)
RUN_ARGS += --run-id "$(RUN_ID)"
endif
RUN_ARGS += --manifest "$(MANIFEST)"
RUN_ARGS += --feature-cpus "$(FEATURE_CPUS)"
RUN_ARGS += --feature-memory "$(FEATURE_MEMORY)"
RUN_ARGS += --feature-time "$(FEATURE_TIME)"

.PHONY: help check doctor preflight build-manifest validate-manifest run submit submit-dry-run smoke clean

help:
	@bash bin/nf1-nextflow-pilot --help

check:
	@bash bin/nf1-nextflow-pilot check

doctor:
	@bash bin/nf1-nextflow-pilot doctor

preflight:
	@bash bin/nf1-nextflow-pilot preflight $(if $(RUN_ID),--run-id "$(RUN_ID)") --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)"

build-manifest:
	@python3 scripts/build_manifest.py $(MANIFEST_ARGS) --output "$(MANIFEST)"

validate-manifest:
	@python3 scripts/build_manifest.py --validate-only --output "$(MANIFEST)"

run:
	@NF1_PILOT_ACCOUNT="$(ACCOUNT)" NF1_PILOT_PARTITION="$(PARTITION)" NF1_PILOT_QOS="$(QOS)" \
	  bash bin/nf1-nextflow-pilot run --profile "$(PROFILE)" $(RUN_ARGS) --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)"

submit:
	@bash bin/nf1-nextflow-pilot submit $(RUN_ARGS) --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)" $(if $(SUBMIT_HOST),--submit-host "$(SUBMIT_HOST)")

submit-dry-run:
	@bash bin/nf1-nextflow-pilot submit $(RUN_ARGS) --account "$(ACCOUNT)" --partition "$(PARTITION)" --qos "$(QOS)" $(if $(SUBMIT_HOST),--submit-host "$(SUBMIT_HOST)") --dry-run

smoke:
	@bash bin/nf1-nextflow-pilot smoke $(if $(RUN_ID),--run-id "$(RUN_ID)")

clean:
	@printf 'Generated pilot artifacts live under 3a.nextflow_pilot/results/ and are gitignored.\n'

