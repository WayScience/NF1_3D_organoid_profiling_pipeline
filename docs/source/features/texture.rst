===========================================================
Texture Features
===========================================================

Description
=================================

Texture features quantify spatial patterns and local intensity variations
within segmented objects. These features are computed from the Gray-Level
Co-occurrence Matrix (GLCM).

Calculation Method
=========================================================================

Texture features are derived from the Gray-Level Co-occurrence Matrix (GLCM),
which captures the frequency of intensity pair relationships at specified offsets.

Parameters
--------------------------------------------------

* **Gray Levels**: 256 (quantization of intensity values)
* **Offset**: 1 voxel (distance for co-occurrence pairs)

These parameters can be adjusted to capture texture patterns at different scales.

Features Extracted
==============================================================

.. list-table:: Texture Feature Measurements
   :header-rows: 1
   :widths: 20 25 12 20 23

   * - Feature
     - Description
     - Range
     - Low Value Indicates
     - High Value Indicates
   * - Angular.Second.Moment
     - Textural uniformity / energy
     - [0, 1]
     - Heterogeneous, varied texture (many different patterns)
     - Uniform, homogeneous texture (repetitive patterns)
   * - Contrast
     - Local variation in intensity
     - [0, ∞)
     - Smooth, low contrast (blurred edges, gradual transitions)
     - Sharp edges, high contrast (crisp boundaries, distinct structures)
   * - Correlation
     - Linear dependency of neighboring gray levels
     - [-1, 1]
     - Uncorrelated, random texture (independent pixel intensities)
     - Correlated texture (organized structures, smooth gradients)
   * - Variance
     - Spread of GLCM intensity values
     - [0, (255)²]
     - Narrow intensity range (uniform brightness)
     - Wide intensity range (high dynamic range)
   * - Inverse.Difference.Moment
     - Local homogeneity
     - [0, 1]
     - Inhomogeneous (varied local intensities)
     - Homogeneous (similar neighboring pixels)
   * - Sum.Average
     - Weighted mean intensity
     - [0, 2(255)]
     - Dark co-occurrence (low signal, dim staining)
     - Bright co-occurrence (strong signal, bright staining)
   * - Sum.Variance
     - Variance of combined intensities
     - [0, ∞)
     - Consistent combined intensities
     - Wide range of intensity sums
   * - Sum.Entropy
     - Randomness in intensity sums
     - [0, log₂(511)] - (Note: 511 = 2*256 - 1)
     - Ordered, predictable sum patterns
     - Random, disordered intensity sums
   * - Entropy
     - Overall texture complexity
     - [0, log₂(65536)] - (Note: 65536 = 256²)
     - Simple, ordered texture (smooth regions, uniform areas)
     - Complex, random texture (highly textured, irregular patterns)
   * - Difference.Variance
     - Variance in intensity differences
     - [0, (65536)] - (Note: 65536 = 256²)
     - Consistent contrast throughout
     - Variable local contrast (mixed smooth/textured regions)
   * - Difference.Entropy
     - Randomness in intensity differences
     - [0, log₂(256)]
     - Smooth transitions (gradual edges)
     - Irregular transitions (fragmented structures, noisy edges)
   * - Info.Measure.Corr.1
     - Mutual information correlation 1
     - [-1, 1]
     - Weak correlation (think random noise)
     - Strong nonlinear correlation (organized patterns)
   * - Info.Measure.Corr.2
     - Mutual information correlation 2
     - [0, 1]
     - Low mutual information (uncorrelated texture)
     - High mutual information (highly organized texture)

**Note:** 255 is used but can be replaced with Ng where:
Ng represents the number of gray levels (256 in this implementation).
