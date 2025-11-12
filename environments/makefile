# Find environment YAMLs (this dir and one level down)
YAML_FILES := $(wildcard *.yml *.yaml */*.yml */*.yaml)

.PHONY: all help list apply apply-all install-local

# Default: show help
all: help

help:
	@echo "Usage:"
	@echo "  make list                # list discovered YAML environment files"
	@echo "  make apply FILE=path.yml # create or update environment from single YAML"
	@echo "  make install-all         # create all discovered YAMLs"
	@echo "  make update-all          # update all discovered YAMLs"

list:
	@sh -c 'if [ -z "$(YAML_FILES)" ]; then echo "No YAML files found."; else printf "%s\n" $(YAML_FILES); fi'

# Apply a single YAML (pass FILE=path/to/env.yml)
apply:
	@if [ -z "$(FILE)" ]; then \
        echo "Specify FILE=path/to/env.yml"; exit 1; \
	fi
	@sh -c 'f="$(FILE)"; if [ ! -f "$$f" ]; then echo "File $$f not found"; exit 1; fi; \
    name=$$(awk "/^[[:space:]]*name:[[:space:]]*/{print $$2; exit}" $$f); \
    if [ -z "$$name" ]; then echo "No 'name:' in $$f; abort"; exit 1; fi; \
    if conda env list | awk "{print $$1}" | grep -qx "$$name"; then \
        echo "Updating $$name from $$f"; mamba env update -f "$$f"; \
    else \
        echo "Creating $$name from $$f"; mamba env create -f "$$f"; \
    fi'

# Apply all discovered YAML files
install-all:
	@if [ -z "$(YAML_FILES)" ]; then \
		echo "No YAML files found."; \
		exit 0; \
	fi; \
	for f in $(YAML_FILES); do \
		name=$$(awk '/^[[:space:]]*name:[[:space:]]*/{print $$2; exit}' "$$f"); \
		if [ -z "$$name" ]; then \
			echo "Skipping $$f (no name)"; \
			continue; \
		fi; \
		if conda env list | awk '{print $$1}' | grep -qx "$$name"; then \
			echo "$$name already exists, skipping creation."; \
		else \
			echo "Creating $$name from $$f"; \
			mamba env create -f "$$f"; \
		fi; \
		if echo "$$name" | grep -q "featurizers"; then \
			echo "Installing additional local featurization_utils for $$name"; \
			conda run -n "$$name" --no-capture-output pip install -e ../src/featurization_utils; \
		fi; \
		if echo "$$name" | grep -q "segmentation"; then \
			echo "Installing additional segmentation_utils for $$name"; \
			conda run -n "$$name" --no-capture-output pip install -e ../src/segmentation_utils; \
		fi; \
		conda run -n "$$name" --no-capture-output pip install -e ../src/file_utils; \
	done

update-all:
	@if [ -z "$(YAML_FILES)" ]; then \
		echo "No YAML files found."; \
		exit 0; \
	fi; \
	for f in $(YAML_FILES); do \
		name=$$(awk '/^[[:space:]]*name:[[:space:]]*/{print $$2; exit}' "$$f"); \
		if [ -z "$$name" ]; then \
			echo "Skipping $$f (no name)"; \
			continue; \
		fi; \
		if conda env list | awk '{print $$1}' | grep -qx "$$name"; then \
			echo "Updating $$name from $$f"; \
			mamba env update -f "$$f"; \
		else \
			echo "Creating $$name from $$f"; \
			mamba env create -f "$$f"; \
		fi; \
	done
