=====================================================================
Colocalization Features
=====================================================================


Description
=========================================================================


Colocalization features quantify the degree to which two different spectral
channels spatially co-occur within a segmented object. These measurements
assess the relationship and correlation between different protein markers.

Overview
=========================================================================


Colocalization features are computed for each unique pair of channels
within the same segmented object.

Features Extracted
=========================================================================


Correlation-Based Metrics
-----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Feature
     - Description
   * - MEAN.CORRELATION.COEFF
     - Mean Pearson correlation coefficient
   * - MEDIAN.CORRELATION.COEFF
     - Median Pearson correlation coefficient
   * - MIN.CORRELATION.COEFF
     - Minimum correlation coefficient
   * - MAX.CORRELATION.COEFF
     - Maximum correlation coefficient

Manders Coefficients (M1 and M2)
------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Feature
     - Description
   * - MEAN.MANDERS.COEFF.M1
     - Average fraction of channel 1 overlapping with channel 2
   * - MEDIAN.MANDERS.COEFF.M1
     - Median M1 coefficient
   * - MIN/MAX.MANDERS.COEFF.M1
     - Min/max M1 values
   * - MEAN.MANDERS.COEFF.M2
     - Average fraction of channel 2 overlapping with channel 1
   * - MEDIAN.MANDERS.COEFF.M2
     - Median M2 coefficient
   * - MIN/MAX.MANDERS.COEFF.M2
     - Min/max M2 values

Overlap Coefficient
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Feature
     - Description
   * - MEAN.OVERLAP.COEFF
     - Average overlap coefficient
   * - MEDIAN.OVERLAP.COEFF
     - Median overlap coefficient
   * - MIN/MAX.OVERLAP.COEFF
     - Min/max overlap coefficients

K1 and K2 Coefficients
-------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Feature
     - Description
   * - MEAN.K1
     - Average intensity correlation quotient for channel 1
   * - MEDIAN.K1
     - Median K1 coefficient
   * - MIN/MAX.K1
     - Min/max K1 values
   * - MEAN.K2
     - Average intensity correlation quotient for channel 2
   * - MEDIAN.K2
     - Median K2 coefficient
   * - MIN/MAX.K2
     - Min/max K2 values

Use Cases
========================================================================

Colocalization features are useful for:

* Determining protein-protein interactions
* Assessing subcellular localization patterns
* Identifying co-distributed cellular components
* Quantifying marker overlap in immunofluorescence

