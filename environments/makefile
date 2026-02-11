# Find all YAML files that match the pattern
YAML_FILES := $(wildcard *glob*.yml * *glob*.yaml *)

# Extract environment names from first line of each YAML (line starting with "name: envname")
ENV_NAMES := $(shell for f in $(YAML_FILES); do head -n 1 $$f | cut -d ' ' -f 2; done)

# Default target: create/update all envs
all: $(ENV_NAMES)

# Rule to create/update each environment by name
%:
	@echo "=== Processing environment $@ ==="
	@if conda env list | grep -q "^$@ "; then \
		echo "Updating environment $@"; \
		mamba env update -f $$(grep -l "name: $@" $(YAML_FILES)); \
	else \
		echo "Creating environment $@"; \
		mamba env create -f $$(grep -l "name: $@" $(YAML_FILES)); \
	fi

	@echo "Installing utils for $@"
	@. $$(conda info --base)/etc/profile.d/conda.sh && conda activate $@ && \
	pip install -e ../utils/src/ && conda deactivate

.PHONY: all $(ENV_NAMES)
