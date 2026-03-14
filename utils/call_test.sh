#!/bin/bash


rm -rf .venv-test
uv venv .venv-test --python 3.12
# shellcheck disable=SC1091
source .venv-test/bin/activate && uv pip install -e ".[dev]"
# Call run_tests.py with flags
python3 tests/run_tests.py

