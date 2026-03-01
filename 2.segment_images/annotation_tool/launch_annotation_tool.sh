#!/bin/bash

conda activate annotation-tool

python src/annotation_tool.py "../data/compressed_images/"

conda deactivate
