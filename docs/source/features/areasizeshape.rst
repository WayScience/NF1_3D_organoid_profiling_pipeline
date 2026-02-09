===============================================================
Area.Size.Shape Features
===============================================================

Description
===============================================

Area.Size.Shape features characterize the geometric properties and spatial
extent of segmented objects. These measurements are computed using both mesh
and voxel-based approaches.

Features Extracted
======================================================

.. list-table:: Area.Size.Shape Feature Measurements
   :header-rows: 1
   :widths: 40 60

   * - Feature
     - Description
   * - VOLUME
     - Total volume of the segmented object in voxels
   * - CENTER.X
     - X-coordinate of object centroid
   * - CENTER.Y
     - Y-coordinate of object centroid
   * - CENTER.Z
     - Z-coordinate of object centroid
   * - BBOX.VOLUME
     - Volume of the bounding box containing the object
   * - MIN.X / MAX.X
     - Minimum and maximum X coordinates
   * - MIN.Y / MAX.Y
     - Minimum and maximum Y coordinates
   * - MIN.Z / MAX.Z
     - Minimum and maximum Z coordinates
   * - EXTENT
     - Ratio of object volume to bounding box volume
   * - EULER.NUMBER
     - Euler characteristic (topological descriptor)
   * - EQUIVALENT.DIAMETER
     - Diameter of a sphere with equivalent volume
   * - SURFACE.AREA
     - Total surface area of the 3D object

Calculation Method
======================================================

These features use 3D mesh reconstruction and voxel analysis:

1. **Mesh Generation**: Create 3D surface mesh from voxel boundaries
2. **Volume Calculation**: Sum voxels within segmented region
3. **Surface Analysis**: Compute surface area from mesh triangulation
4. **Spatial Statistics**: Calculate centroid and bounding box properties

Applications
================================================

Area.Size.Shape features are useful for:

* Quantifying cell and organoid growth
* Detecting morphological abnormalities
* Comparing size distributions across conditions
* Identifying fusion or fragmentation events
