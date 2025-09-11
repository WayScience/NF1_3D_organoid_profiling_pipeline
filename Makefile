# Makefile
GIT_ROOT := $(shell git rev-parse --show-toplevel)

default:
	@echo $(GIT_ROOT)
