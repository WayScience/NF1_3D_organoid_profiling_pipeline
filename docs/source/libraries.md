---
Title: libraries
---

# Python scientific stack

The feature extraction pipeline leverages industry-standard python scientific libraries for efficient image processing and computation.

## Core libraries

### Scikit-image

**Purpose**: image processing and analysis
**Website**: <https://scikit-image.org/>

Features used:

- Image segmentation algorithms
- Morphological operations
- 3D image analysis
- Regional property measurement

### Scipy

**Purpose**: scientific computing
**Website**: <https://scipy.org/>

Features used:

- Advanced signal processing
- Statistical distributions
- Interpolation and filtering
- Multi-dimensional image processing

### Mahotas

**Purpose**: image analysis and processing
**Website**: <https://mahotas.readthedocs.io/en/latest>

Features used:

- Gray-level co-occurrence matrix (GLCM)
- Texture feature extraction
- Morphological operations

### Numpy

**Purpose**: numerical computing
**Website**: <https://numpy.org/>

Features used:

- Multi-dimensional array operations
- Vectorized computations
- Statistical calculations

## GPU-accelerated libraries

### Cupy

**Purpose**: GPU-accelerated arrays (drop-in numpy replacement)
**Website**: <https://docs.cupy.dev/en/stable>

Features:

- GPU memory management
- Parallel numerical operations
- 10-100X speedup over CPU numpy

### Cucim

**Purpose**: GPU-accelerated image processing
**Website**: <https://docs.rapids.ai/api/cucim/stable/>

Features:

- GPU-accelerated scikit-image operations
- Fast morphological operations
- Efficient image filtering
