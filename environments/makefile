# Find all YAML files (adjust patterns as needed)
YAML_FILES := $(wildcard *.yml *.yaml)

# Extract environment names from YAML files (line starting with "name: ")
ENV_NAMES := $(shell for f in $(YAML_FILES); do \
	grep "^name:" $$f | head -n 1 | sed 's/^name:[[:space:]]*//'; \
done)

# Debug: show what we found
$(info YAML_FILES: $(YAML_FILES))
$(info ENV_NAMES: $(ENV_NAMES))

# Default target: create/update all envs
all: $(ENV_NAMES)
	@echo "$(ENV_NAMES) environments created/updated successfully."

# Update target: force update all existing environments
update: $(addprefix update-,$(ENV_NAMES))
	@echo "$(ENV_NAMES) environments updated successfully."

# Rule to create/update each environment by name
$(ENV_NAMES):
	@echo "=== Processing environment $@ ==="
	@if conda env list | grep -q "^$@ "; then \
		echo "Updating environment $@"; \
		mamba env update -f $$(grep -l "name:[[:space:]]*$@" $(YAML_FILES)); \
	else \
		echo "Creating environment $@"; \
		mamba env create -f $$(grep -l "name:[[:space:]]*$@" $(YAML_FILES)); \
	fi
	@echo "Installing utils for $@"
	@. $$(conda info --base)/etc/profile.d/conda.sh && conda activate $@ && \
		pip install -e ../utils/ && conda deactivate

# Rule to force update a specific environment
update-%:
	@echo "=== Force updating environment $* ==="
	@mamba env update -f $$(grep -l "name:[[:space:]]*$*" $(YAML_FILES))
	@echo "Reinstalling utils for $*"
	@. $$(conda info --base)/etc/profile.d/conda.sh && conda activate $* && \
		pip install -e ../utils/src/ && conda deactivate

.PHONY: all update $(ENV_NAMES) $(addprefix update-,$(ENV_NAMES))
