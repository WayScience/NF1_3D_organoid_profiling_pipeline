#!/bin/bash

view=True

conda activate NF1_3D_featurization_sphinx_env

make clean
make html

conda deactivate

if [ "$view" = True ] ; then
    firefox build/html/index.html
fi
