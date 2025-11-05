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

	@if [ "$@" = "featurizer" ] || [ "$@" = "deep_learning_featurizer" ]; then \
		echo "Installing featurization_utils in $@"; \
		cd .. && . $$(conda info --base)/etc/profile.d/conda.sh && conda activate $@ && \
		pip install -e ./src/featurization_utils && conda deactivate; \
	fi

	@echo "Installing file_utils for $@"
	@. $$(conda info --base)/etc/profile.d/conda.sh && conda activate $@ && \
	pip install -e ./src/file_utils && conda deactivate

.PHONY: all $(ENV_NAMES)
