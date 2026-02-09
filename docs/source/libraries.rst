=========
Libraries
=========

Python Scientific Stack
=======================

The feature extraction pipeline leverages industry-standard Python
scientific libraries for efficient image processing and computation.

Core Libraries
==============

scikit-image
------------

**Purpose**: Image processing and analysis
**Website**: https://scikit-image.org/

Features used:

* Image segmentation algorithms
* Morphological operations
* 3D image analysis
* Regional property measurement

scipy
-----

**Purpose**: Scientific computing
**Website**: https://www.scipy.org/

Features used:

* Advanced signal processing
* Statistical distributions
* Interpolation and filtering
* Multi-dimensional image processing

mahotas
-------

**Purpose**: Image analysis and processing
**Website**: https://mahotas.readthedocs.io/

Features used:

* Gray-level co-occurrence matrix (GLCM)
* Texture feature extraction
* Morphological operations

numpy
-----

**Purpose**: Numerical computing
**Website**: https://numpy.org/

Features used:

* Multi-dimensional array operations
* Vectorized computations
* Statistical calculations

GPU-Accelerated Libraries
==========================

cupy
----

**Purpose**: GPU-accelerated arrays (drop-in NumPy replacement)
**Website**: https://docs.cupy.dev/

Features:

* GPU memory management
* Parallel numerical operations
* 10-100x speedup over CPU NumPy

cucim
-----

**Purpose**: GPU-accelerated image processing
**Website**: https://docs.rapids.ai/api/cucim/

Features:

* GPU-accelerated scikit-image operations
* Fast morphological operations
* Efficient image filtering

