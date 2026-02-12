#!/bin/bash

conda activate annotation-tool

python annotation_tool.py "../data/compressed_images/"

conda deactivate
