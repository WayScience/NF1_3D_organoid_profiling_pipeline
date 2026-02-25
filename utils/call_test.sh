#!/bin/bash

uv venv .venv-test --python 3.13
# shellcheck disable=SC1091
source .venv-test/bin/activate && uv pip install -e ".[dev]"
# Call run_tests.py with flags
cd tests || exit
python3 run_tests.py
cd ../  || exit
