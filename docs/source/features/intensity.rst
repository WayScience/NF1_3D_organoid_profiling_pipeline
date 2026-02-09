=======================================================================
Intensity Features
=======================================================================

Description
=================================================================

Intensity features quantify the pixel/voxel value distributions within
segmented objects. These measurements capture both overall intensity levels
and spatial intensity patterns.

Features Extracted
========================================================================

Location-Based Intensity
----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - CM.X / CM.Y / CM.Z
     - Average intensity-weighted coordinates in each spatial dimension
   * - CMI.X / CMI.Y / CMI.Z
     - Center of mass intensity in each spatial dimension
   * - I.X / I.Y / I.Z
     - Integrated intensity along each axis
   * - MAX.X / MAX.Y / MAX.Z
     - Coordinates of maximum intensity

Statistical Measures
------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - MEAN.INTENSITY
     - Average intensity within object
   * - MEDIAN.INTENSITY
     - Median intensity value
   * - MAX.INTENSITY
     - Maximum intensity value
   * - MIN.INTENSITY
     - Minimum intensity value
   * - STD.INTENSITY
     - Standard deviation of intensity
   * - MAD.INTENSITY
     - Median absolute deviation
   * - LOWER.QUARTILE.INTENSITY
     - 25th percentile
   * - UPPER.QUARTILE.INTENSITY
     - 75th percentile

Edge-Based Measurements
---------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Description
   * - INTEGRATED.INTENSITY.EDGE
     - Sum of edge pixel intensities
   * - MEAN.INTENSITY.EDGE
     - Average edge intensity
   * - MAX.INTENSITY.EDGE
     - Maximum edge intensity
   * - MIN.INTENSITY.EDGE
     - Minimum edge intensity
   * - STD.INTENSITY.EDGE
     - Standard deviation of edge intensities
   * - EDGE.COUNT
     - Number of edge voxels

Other Measurements
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - VOLUME
     - Object volume (included for reference)
   * - DIFF.X / DIFF.Y / DIFF.Z
     - Spatial intensity gradient in each dimension
   * - MASS.DISPLACEMENT
     - Distance between geometric and intensity centers

Calculation Method
======================================================

Intensity features are computed as follows:

1. **Segmentation Masking**: Apply object masks to isolate pixel/voxel values
2. **Statistical Computation**: Calculate mean, median, max, min, std, etc.
3. **Spatial Analysis**: Determine intensity-weighted coordinates and gradients
4. **Edge Detection**: Identify edge pixels/voxels and compute edge statistics

Applications
===================================================

Intensity features are useful for:

* Quantifying fluorescence levels within cellular structures
* Detecting changes in protein expression
* Characterizing subcellular localization patterns
* Identifying phenotypic alterations due to treatments
