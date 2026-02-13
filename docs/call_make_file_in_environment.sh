#!/bin/bash

view=True

conda activate NF1_3D_featurization_sphinx_env

make clean
make html

conda deactivate

if [ "$view" = True ] ; then
    if command -v open >/dev/null 2>&1; then
        open build/html/index.html
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open build/html/index.html
    elif command -v firefox >/dev/null 2>&1; then
        firefox build/html/index.html
    else
        echo "No supported browser opener found. Open build/html/index.html manually."
    fi
fi
