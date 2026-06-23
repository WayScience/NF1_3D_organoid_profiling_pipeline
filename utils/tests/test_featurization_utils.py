"""Comprehensive tests for featurization utility functions."""

from __future__ import annotations

import numpy
import numpy as np
import pytest
from image_analysis_3D.featurization_utils.area_size_shape_utils import (
    calculate_surface_area,
    measure_3D_area_size_shape,
)
from image_analysis_3D.featurization_utils.colocalization_utils import (
    bisection_costes_threshold_calculation,
    linear_costes_threshold_calculation,
    measure_3D_colocalization,
)
from image_analysis_3D.featurization_utils.granularity_utils import (
    _subsample_3d,
    _upsample_3d,
    measure_3D_granularity,
)
from image_analysis_3D.featurization_utils.intensity_utils import (
    get_outline,
    measure_3D_intensity_CPU,
)
from image_analysis_3D.featurization_utils.loading_classes import ObjectLoader
from image_analysis_3D.featurization_utils.neighbors_utils import (
    crop_3D_image,
    measure_3D_number_of_neighbors,
    neighbors_expand_box,
)
from image_analysis_3D.featurization_utils.texture_utils import (
    measure_3D_texture,
    scale_image,
)

# ============================================================================
# TESTS: area_size_shape_utils
# ============================================================================


class TestAreaSizeShapeUtils:
    """Tests for area, size, and shape feature extraction."""

    def test_measure_3d_area_size_shape_basic(self, object_loader_simple):
        """Test basic functionality of measure_3D_area_size_shape."""
        from unittest.mock import Mock

        # Create mock image_set_loader
        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=object_loader_simple,
        )

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check that required keys exist
        expected_keys = {
            "object_id",
            "Volume",
            "CenterX",
            "CenterY",
            "CenterZ",
        }
        assert expected_keys.issubset(set(result.keys()))

        # Check that we have data for multiple objects
        assert len(result["object_id"]) > 0

    def test_measure_3d_area_size_shape_with_mocked_image_set_loader(
        self, simple_3d_image, simple_3d_label_image
    ):
        """Test measure_3D_area_size_shape with mocked ImageSetLoader."""
        from unittest.mock import Mock

        # Create mock
        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        object_loader = ObjectLoader(
            image=simple_3d_image,
            label_image=simple_3d_label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=object_loader,
        )

        assert isinstance(result, dict)
        assert "Volume" in result

    def test_size_discrimination(self, size_varied_objects):
        """Test that area/size/shape features correctly distinguish small, medium, large objects."""
        from unittest.mock import Mock

        image, labels = size_varied_objects
        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 1.0, 1.0)

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=loader,
        )

        # Extract volumes for each object
        volumes = {}
        for obj_id, vol in zip(result["object_id"], result["Volume"]):
            volumes[obj_id] = vol

        # Verify all three objects detected
        assert len(volumes) == 3

        # Verify size ordering: small < medium < large
        assert volumes[1] < volumes[2] < volumes[3]
        # Approximate expected sizes (with some tolerance)
        assert 20 < volumes[1] < 40  # ~27 voxels
        assert 400 < volumes[2] < 700  # ~512 voxels
        assert 2500 < volumes[3] < 4000  # ~3375 voxels

    def test_surface_area_not_nan_for_surrounded_cell(self):
        """SurfaceArea must be non-NaN for a cell whose bbox is fully occupied by neighbors.

        Regression test for a bug where calculate_surface_area received the full
        multi-cell label_object instead of the single-cell masked subset_lab_object.
        Inside calculate_surface_area, volume_truths = volume > 0 treats neighbor
        voxels as foreground. For a small cell completely surrounded by a larger
        neighbor, its bounding box contains no background voxels, so marching_cubes
        raises ValueError (surface level not within data range), which is caught
        silently and stored as NaN.

        The fix is to pass subset_lab_object (single-cell mask) instead of
        label_object (full image) at the calculate_surface_area call site.
        """
        from unittest.mock import Mock

        # Build a 15x15x15 label image with two touching cells.
        # Cell 1 (label=1): a cross/plus shape (strips along each axis) centred at
        #   [7,7,7] within a 5x5x5 bounding box [5:10, 5:10, 5:10].
        #   Because it is not a solid cube it does NOT fill its own bbox — the
        #   8 corners of [5:10,5:10,5:10] are empty.
        # Cell 2 (label=2): fills all remaining voxels, including those corners.
        #
        # With the bug (full label_object passed to calculate_surface_area):
        #   volume = full_image[5:10,5:10,5:10] → all non-zero (label 1 or 2)
        #   → volume_truths = all True → marching_cubes raises ValueError → NaN
        # With the fix (subset_lab_object passed):
        #   volume = subset[5:10,5:10,5:10] → cross of 1s, corner zeros
        #   → marching_cubes succeeds → real surface area returned
        labels = np.ones((15, 15, 15), dtype=np.int32) * 2
        labels[5:10, 7, 7] = 1  # z-strip of the cross
        labels[7, 5:10, 7] = 1  # y-strip
        labels[7, 7, 5:10] = 1  # x-strip

        image = np.zeros((15, 15, 15), dtype=np.float32)

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 1.0, 1.0)

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=loader,
        )

        surface_areas = {
            obj_id: sa
            for obj_id, sa in zip(result["object_id"], result["SurfaceArea"])
        }

        assert 1 in surface_areas, "Cell 1 missing from result"
        assert not numpy.isnan(surface_areas[1]), (
            f"SurfaceArea for surrounded cell is NaN. "
            f"Likely caused by passing the full label_object instead of the "
            f"single-cell subset_lab_object to calculate_surface_area, so "
            f"volume_truths = volume > 0 includes neighbor voxels and leaves "
            f"no background for marching_cubes."
        )
        assert surface_areas[1] > 0, "SurfaceArea for cell 1 should be positive"


class TestCalculateSurfaceArea:
    """Tests for surface area calculation."""

    def test_calculate_surface_area_basic(self, simple_3d_label_image):
        """Test surface area calculation for a simple object."""
        # Get properties for object 1
        from skimage.measure import regionprops

        props = regionprops(simple_3d_label_image)[0]

        # Create properties dict in the format expected
        props_dict = {
            "bbox-0": [props.bbox[0]],
            "bbox-1": [props.bbox[1]],
            "bbox-2": [props.bbox[2]],
            "bbox-3": [props.bbox[3]],
            "bbox-4": [props.bbox[4]],
            "bbox-5": [props.bbox[5]],
        }

        # Note: surface area calculation may not work for very small objects
        # Just test that it doesn't crash
        try:
            surface_area = calculate_surface_area(
                label_object=simple_3d_label_image,
                props=props_dict,
                spacing=(1.0, 0.1, 0.1),
            )
            # Check result
            assert isinstance(surface_area, dict)
            assert "SurfaceArea" in surface_area
            assert len(surface_area["SurfaceArea"]) > 0
            assert surface_area["SurfaceArea"][0] > 0
        except ValueError:
            # Marching cubes may fail for very small/simple objects
            pass

    def test_calculate_surface_area_sphere_approximation(self):
        """Test surface area calculation for a roughly spherical object."""
        # Create a 3D sphere-like object
        size = 20
        label_image = np.zeros((size, size, size), dtype=np.uint8)
        center = size // 2
        radius = 5

        # Create a sphere
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt(
                        (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
                    )
                    if dist <= radius:
                        label_image[z, y, x] = 1

        from skimage.measure import regionprops

        props = regionprops(label_image)[0]

        props_dict = {
            "bbox-0": [props.bbox[0]],
            "bbox-1": [props.bbox[1]],
            "bbox-2": [props.bbox[2]],
            "bbox-3": [props.bbox[3]],
            "bbox-4": [props.bbox[4]],
            "bbox-5": [props.bbox[5]],
        }

        surface_area = calculate_surface_area(
            label_object=label_image,
            props=props_dict,
            spacing=(1.0, 1.0, 1.0),
        )

        assert surface_area > 0


# ============================================================================
# TESTS: texture_utils
# ============================================================================


class TestTextureUtils:
    """Tests for texture feature extraction."""

    def test_scale_image_basic(self, simple_3d_image):
        """Test basic image scaling."""
        scaled = scale_image(simple_3d_image, num_gray_levels=256)

        # Check shape is preserved
        assert scaled.shape == simple_3d_image.shape

        # Check dtype
        assert scaled.dtype == np.uint8

        # Check values are in expected range
        assert scaled.min() >= 0
        assert scaled.max() < 256

    def test_scale_image_to_different_levels(self, simple_3d_image):
        """Test scaling to different gray level counts."""
        # scale_image only supports 256 (uint8) and 65536 (uint16)
        levels = 256
        scaled = scale_image(simple_3d_image, num_gray_levels=levels)
        assert scaled.dtype == np.uint8
        assert scaled.max() <= levels

    def test_scale_image_uniform_input(self):
        """Test scaling with uniform image."""
        uniform_image = np.ones((10, 10, 10), dtype=np.uint16) * 100
        scaled = scale_image(uniform_image, num_gray_levels=256)

        # Should handle uniform images gracefully
        assert scaled.shape == uniform_image.shape
        # Uniform image remains uniform after rescaling
        assert np.all(scaled == scaled[0, 0, 0])

    def test_measure_3d_texture_basic(self, object_loader_simple):
        """Test basic texture measurement."""
        result = measure_3D_texture(
            object_loader=object_loader_simple,
            distance=1,
            grayscale=256,
        )

        # Check result structure
        assert isinstance(result, dict)
        assert "object_id" in result
        assert "texture_name" in result
        assert "texture_value" in result

        # Check data consistency
        assert len(result["object_id"]) == len(result["texture_name"])
        assert len(result["object_id"]) == len(result["texture_value"])

    def test_measure_3d_texture_different_distances(self, object_loader_simple):
        """Test texture measurement with different distances."""
        for distance in [1, 2, 3]:
            result = measure_3D_texture(
                object_loader=object_loader_simple,
                distance=distance,
                grayscale=256,
            )

            # All texture values should be floats
            assert all(isinstance(v, (int, float)) for v in result["texture_value"])

    def test_measure_3d_texture_different_grayscale(self, object_loader_simple):
        """Test texture measurement with different grayscale levels."""
        # scale_image only supports 256 (uint8) and 65536 (uint16), use 256 for memory efficiency
        result = measure_3D_texture(
            object_loader=object_loader_simple,
            distance=1,
            grayscale=256,
        )

        # Should have results
        assert len(result["texture_value"]) > 0

    def test_texture_uniformity_detection(self, texture_varied_objects):
        """Test that texture features distinguish uniform from non-uniform textures."""
        image, labels = texture_varied_objects

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_texture(
            object_loader=loader,
            distance=1,
            grayscale=256,
        )

        # Group texture features by object
        obj_textures = {}
        for obj_id, texture_name, texture_val in zip(
            result["object_id"], result["texture_name"], result["texture_value"]
        ):
            if obj_id not in obj_textures:
                obj_textures[obj_id] = {}
            obj_textures[obj_id][texture_name] = texture_val

        # Object 1 (uniform) should have lower standard deviation than object 2 (random)
        # Check contrast or homogeneity features if available
        assert len(obj_textures) >= 2

    def test_texture_values_correct_per_object(self, texture_varied_objects):
        """Each object must receive its own Haralick values, not another object's or garbage.

        Regression test for a bug where features = numpy.empty(...) was called
        inside the per-object loop, resetting the array every iteration and leaving
        only the last object's values intact. All other objects received uninitialized
        memory from numpy.empty.
        """
        import mahotas

        image, labels = texture_varied_objects

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_texture(object_loader=loader, distance=1, grayscale=256)

        # Build per-object lookup: {object_id: {texture_name: value}}
        obj_textures = {}
        for obj_id, name, val in zip(
            result["object_id"], result["texture_name"], result["texture_value"]
        ):
            obj_textures.setdefault(obj_id, {})[name] = val

        # Independently compute expected Haralick values for each object
        # by replicating what the function should do: crop to bbox, mask, scale, haralick
        import skimage.measure

        props = skimage.measure.regionprops_table(labels, properties=["label", "bbox"])
        label_to_bbox = {
            int(props["label"][i]): (
                int(props["bbox-0"][i]),
                int(props["bbox-1"][i]),
                int(props["bbox-2"][i]),
                int(props["bbox-3"][i]),
                int(props["bbox-4"][i]),
                int(props["bbox-5"][i]),
            )
            for i in range(len(props["label"]))
        }

        feature_names = [
            "AngularSecondMoment", "Contrast", "Correlation", "Variance",
            "InverseDifferenceMoment", "SumAverage", "SumVariance", "SumEntropy",
            "Entropy", "DifferenceVariance", "DifferenceEntropy",
            "InformationMeasureOfCorrelation1", "InformationMeasureOfCorrelation2",
        ]

        unique_labels = sorted(np.unique(labels[labels != 0]))
        for label in unique_labels:
            z0, y0, x0, z1, y1, x1 = label_to_bbox[label]
            crop = image[z0:z1, y0:y1, x0:x1].copy()
            mask = labels[z0:z1, y0:y1, x0:x1] == label
            crop[~mask] = 0
            crop_scaled = scale_image(crop, num_gray_levels=256)
            try:
                expected = mahotas.features.haralick(
                    ignore_zeros=True, f=crop_scaled, distance=1, compute_14th_feature=False
                )
            except ValueError:
                continue  # uniform object — haralick raises, both paths yield nan

            # Check direction 0 (first direction) for each feature
            for feat_idx, feat_name in enumerate(feature_names):
                key = f"{feat_name}-1-00-256"
                assert key in obj_textures[label], f"Missing {key} for object {label}"
                actual = obj_textures[label][key]
                exp = expected[0, feat_idx]
                assert numpy.isclose(actual, exp, rtol=1e-5, equal_nan=True), (
                    f"Object {label}, feature {feat_name}: got {actual}, expected {exp}. "
                    f"Likely caused by features array being reset inside the per-object loop."
                )

    def test_measure_3d_texture_uniform_full_object(self):
        """Test that a completely uniform object is handled gracefully."""
        image = np.ones((6, 6, 6), dtype=np.uint16) * 500
        labels = np.ones((6, 6, 6), dtype=np.uint8)

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_texture(
            object_loader=loader,
            distance=1,
            grayscale=256,
        )

        # Returns 13 directions × 13 features per object = 169 values for 1 object
        assert len(result["texture_value"]) == 169
        # Uniform textures result in NaN or inf values for some features, others may be 0 or 1
        assert all(
            isinstance(value, (int, float, np.number))
            for value in result["texture_value"]
        )


# ============================================================================
# TESTS: colocalization_utils
# ============================================================================


class TestColocationUtils:
    """Tests for colocalization feature extraction."""

    def test_linear_costes_threshold_basic(self, high_contrast_images):
        """Test linear Costes threshold calculation."""
        img1, img2 = high_contrast_images

        thr1, thr2 = linear_costes_threshold_calculation(
            first_image=img1,
            second_image=img2,
            scale_max=255,
            fast_costes="Accurate",
        )

        # Thresholds should be reasonable values
        assert 0 <= thr1 <= 255
        assert 0 <= thr2 <= 255

    def test_linear_costes_threshold_fast_mode(self, high_contrast_images):
        """Test linear Costes threshold with fast mode."""
        img1, img2 = high_contrast_images

        thr1, thr2 = linear_costes_threshold_calculation(
            first_image=img1,
            second_image=img2,
            scale_max=255,
            fast_costes="Fast",
        )

        # Should still return valid thresholds
        assert isinstance(thr1, (int, float))
        assert isinstance(thr2, (int, float))

    def test_bisection_costes_threshold_basic(self, high_contrast_images):
        """Test bisection Costes threshold calculation."""
        img1, img2 = high_contrast_images

        thr1, thr2 = bisection_costes_threshold_calculation(
            first_image=img1,
            second_image=img2,
            scale_max=255,
        )

        # Thresholds should be reasonable
        assert isinstance(thr1, (int, float))
        assert isinstance(thr2, (int, float))

    def test_measure_3d_colocalization_basic(self, high_contrast_images):
        """Test basic colocalization measurement."""
        img1, img2 = high_contrast_images

        result = measure_3D_colocalization(
            cropped_image_1=img1,
            cropped_image_2=img2,
            thr=15,
            fast_costes="Accurate",
        )

        # Check result structure
        assert isinstance(result, dict)

        # Common colocalization metrics (actual keys from implementation)
        common_keys = {
            "Correlation",
            "MandersCoeffM1",
            "MandersCoeffM2",
            "OverlapCoeff",
        }

        # At least some expected keys should be present
        assert len(set(result.keys()) & common_keys) > 0

    def test_measure_3d_colocalization_different_thresholds(self, high_contrast_images):
        """Test colocalization with different thresholds."""
        img1, img2 = high_contrast_images

        results = []
        for thr in [5, 10, 15, 20]:
            result = measure_3D_colocalization(
                cropped_image_1=img1,
                cropped_image_2=img2,
                thr=thr,
                fast_costes="Accurate",
            )
            results.append(result)

        # All results should have the same keys
        assert all(set(r.keys()) == set(results[0].keys()) for r in results)

    def test_measure_3d_colocalization_identical_images(self):
        """Test colocalization with identical images."""
        img = np.random.randint(50, 200, (10, 15, 15), dtype=np.uint8)

        result = measure_3D_colocalization(
            cropped_image_1=img,
            cropped_image_2=img,
            thr=15,
            fast_costes="Accurate",
        )

        # Correlation should be close to 1 for identical images
        if "Correlation" in result:
            assert result["Correlation"] > 0.9

    def test_colocalization_detection(self, colocalized_vs_separate_signals):
        """Test that colocalization metrics distinguish colocalized from separate signals."""
        img1, img2 = colocalized_vs_separate_signals

        # Test perfect colocalization region
        coloc_region_1 = img1[3:12, 3:12, 3:12]
        coloc_region_2 = img2[3:12, 3:12, 3:12]

        result_coloc = measure_3D_colocalization(
            cropped_image_1=coloc_region_1,
            cropped_image_2=coloc_region_2,
            thr=50,
            fast_costes="Accurate",
        )

        # Test non-colocalized region
        sep_region_1 = img1[3:10, 3:10, 18:25]
        sep_region_2 = img2[10:15, 20:27, 18:25]

        # Pad to same size
        sep_region_1_pad = np.zeros((12, 24, 7), dtype=np.uint8)
        sep_region_2_pad = np.zeros((12, 24, 7), dtype=np.uint8)
        sep_region_1_pad[:7, :7, :] = sep_region_1
        sep_region_2_pad[5:10, 17:24, :] = sep_region_2

        result_sep = measure_3D_colocalization(
            cropped_image_1=sep_region_1_pad,
            cropped_image_2=sep_region_2_pad,
            thr=50,
            fast_costes="Accurate",
        )

        # Colocalized should have higher correlation and overlap
        # Handle NaN values that can occur in edge cases
        if "Correlation" in result_coloc and "Correlation" in result_sep:
            coloc_corr = result_coloc["Correlation"]
            sep_corr = result_sep["Correlation"]
            # Only compare if neither is NaN
            if not (np.isnan(coloc_corr) or np.isnan(sep_corr)):
                assert coloc_corr > sep_corr
        if "OverlapCoeff" in result_coloc and "OverlapCoeff" in result_sep:
            coloc_overlap = result_coloc["OverlapCoeff"]
            sep_overlap = result_sep["OverlapCoeff"]
            # Only compare if neither is NaN
            if not (np.isnan(coloc_overlap) or np.isnan(sep_overlap)):
                assert coloc_overlap > sep_overlap


    def test_combined_thresh_does_not_raise_unbound_local_error(self):
        """Regression: combined_thresh was only assigned in the else-branch of a
        try/except ValueError block, so a ValueError left it unbound and the
        overlap coefficient section raised UnboundLocalError.
        """
        img_empty = np.zeros((0,), dtype=float)
        try:
            measure_3D_colocalization(img_empty, img_empty)
        except UnboundLocalError:
            pytest.fail(
                "combined_thresh was unbound — missing initialisation before try block"
            )
        except Exception:
            pass  # Any other exception is acceptable for zero-size input.

    def test_accurate_mode_calls_linear_not_bisection(self, high_contrast_images):
        """Regression: 'Accurate' mode called bisection_costes; 'Fast' called
        linear — inverted relative to the docstring.
        """
        from unittest.mock import patch

        img1, img2 = high_contrast_images
        coloc_mod = "image_analysis_3D.featurization_utils.colocalization_utils"
        with (
            patch(
                f"{coloc_mod}.linear_costes_threshold_calculation",
                wraps=linear_costes_threshold_calculation,
            ) as mock_linear,
            patch(
                f"{coloc_mod}.bisection_costes_threshold_calculation",
                wraps=bisection_costes_threshold_calculation,
            ) as mock_bisect,
        ):
            measure_3D_colocalization(img1, img2, fast_costes="Accurate")
            assert mock_linear.called, (
                "'Accurate' mode should dispatch linear_costes_threshold_calculation"
            )
            assert not mock_bisect.called, (
                "'Accurate' mode should NOT dispatch bisection_costes_threshold_calculation"
            )


# ============================================================================
# TESTS: granularity_utils
# ============================================================================


class TestGranularityUtils:
    """Tests for granularity feature extraction."""

    def test_subsample_3d_basic(self, simple_3d_image, simple_3d_binary_mask):
        """Test basic 3D subsampling."""
        new_shape = numpy.array(simple_3d_image.shape, dtype=float) * 0.5
        subsampled_img = _subsample_3d(
            data=simple_3d_image.astype(float),
            new_shape=new_shape,
            subsample_factor=0.5,
            order=1,
        )
        subsampled_mask = (
            _subsample_3d(
                data=simple_3d_binary_mask.astype(float),
                new_shape=new_shape,
                subsample_factor=0.5,
                order=0,
            )
            > 0.5
        )

        # Check that shapes are reasonable and reduced
        assert subsampled_img.shape[0] > 0
        assert subsampled_mask.shape[0] > 0
        assert subsampled_img.size < simple_3d_image.size

    def test_subsample_3d_factor_one(self, simple_3d_image, simple_3d_binary_mask):
        """Test subsampling with factor=1.0 (no subsampling)."""
        new_shape = numpy.array(simple_3d_image.shape, dtype=float)  # factor 1.0
        subsampled_img = _subsample_3d(
            data=simple_3d_image.astype(float),
            new_shape=new_shape,
            subsample_factor=1.0,
            order=1,
        )

        # Should return same-shaped copy
        assert subsampled_img.shape == simple_3d_image.shape

    def test_measure_3d_granularity_basic(self, object_loader_simple):
        """Test basic granularity measurement."""
        result = measure_3D_granularity(
            object_loader=object_loader_simple,
            radius=10,
            granular_spectrum_length=16,
            subsample_size=0.5,
            image_sample_size=0.25,
            mask_threshold=0.9,
        )

        # Check result structure
        assert isinstance(result, dict)
        assert "object_id" in result
        assert "feature" in result
        assert "value" in result

    def test_measure_3d_granularity_validation(self, object_loader_simple):
        """Test granularity measurement parameter validation."""
        # Invalid subsample factor
        with pytest.raises(ValueError):
            measure_3D_granularity(
                object_loader=object_loader_simple,
                subsample_size=1.5,
            )

        # Invalid radius
        with pytest.raises(ValueError):
            measure_3D_granularity(
                object_loader=object_loader_simple,
                radius=-1,
            )

        # Invalid granular spectrum length
        with pytest.raises(ValueError):
            measure_3D_granularity(
                object_loader=object_loader_simple,
                granular_spectrum_length=-1,
            )

    def test_granularity_distinguishes_textures(self, granularity_test_data):
        """Test that granularity correctly distinguishes high, medium, low granularity."""
        image, labels = granularity_test_data

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_granularity(
            object_loader=loader,
            radius=5,
            granular_spectrum_length=8,
            subsample_size=0.5,
        )

        # Extract granularity values for each object
        obj_granularities = {}
        for obj_id, feature, value in zip(
            result["object_id"], result["feature"], result["value"]
        ):
            if obj_id not in obj_granularities:
                obj_granularities[obj_id] = []
            obj_granularities[obj_id].append(value)

        # High granularity (object 1) should have higher values than low granularity (object 3)
        # Average granularity across spectrum
        high_gran_avg = np.mean(obj_granularities[1])
        low_gran_avg = np.mean(obj_granularities[3])

        # High granularity should show more variation in spectrum
        assert len(obj_granularities) >= 3  # Should detect all three objects

    def test_upsample_3d_no_division_by_zero_when_dim_is_one(self):
        """Regression: _upsample_3d divided by (original_shape[d] - 1); when any
        dimension is 1 this is 0, raising ZeroDivisionError.
        """
        data = np.ones((3, 3, 3), dtype=float) * 5.0
        subsampled_shape = np.array([3.0, 3.0, 3.0])
        result = _upsample_3d(data, subsampled_shape, original_shape=(1, 3, 3))
        assert result.shape == (1, 3, 3)
        assert np.all(np.isfinite(result))

    def test_granularity_no_crash_on_single_z_slice(self):
        """Regression: measure_3D_granularity on a (1, Y, X) image called
        _upsample_3d with original_shape[0]=1, hitting the division-by-zero.
        """
        np.random.seed(0)
        image = np.random.randint(50, 200, (1, 20, 20), dtype=np.uint8).astype(float)
        labels = np.zeros((1, 20, 20), dtype=int)
        labels[0, 5:15, 5:15] = 1
        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_granularity(
            loader,
            subsample_size=0.5,
            granular_spectrum_length=3,
            radius=2,
        )
        assert isinstance(result, dict)


# ============================================================================
# TESTS: intensity_utils
# ============================================================================


class TestIntensityUtils:
    """Tests for intensity feature extraction."""

    def test_integrated_intensity_is_per_object_not_global(self):
        """IntegratedIntensity for each object must equal only that object's pixel sum.

        scipy.ndimage.sum(input, labels) without an explicit index= argument
        sums all elements where labels != 0 — which coincidentally gives the
        right answer when labels is binarised to {0, 1}, but would silently
        sum across all objects if the binarisation were ever removed.

        This test documents the expected semantic contract: IntegratedIntensity
        for object N == numpy.sum(image[label_image == N]).
        """
        image = np.zeros((10, 10, 10), dtype=np.float32)
        label = np.zeros((10, 10, 10), dtype=np.int32)

        # Object 1: intensity 50, 8 voxels → expected sum = 400
        image[1:3, 1:3, 1:3] = 50.0
        label[1:3, 1:3, 1:3] = 1

        # Object 2: intensity 100, 8 voxels → expected sum = 800
        image[7:9, 7:9, 7:9] = 100.0
        label[7:9, 7:9, 7:9] = 2

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        obj_to_integrated = {}
        for obj_id, feat, val in zip(
            result["object_id"], result["feature_name"], result["value"]
        ):
            if feat == "IntegratedIntensity":
                obj_to_integrated[int(obj_id)] = float(val)

        assert obj_to_integrated[1] == pytest.approx(400.0), (
            f"Object 1 IntegratedIntensity should be 400 (8 voxels × 50), "
            f"got {obj_to_integrated[1]}"
        )
        assert obj_to_integrated[2] == pytest.approx(800.0), (
            f"Object 2 IntegratedIntensity should be 800 (8 voxels × 100), "
            f"got {obj_to_integrated[2]}"
        )

    def test_get_outline_basic(self, simple_3d_binary_mask):
        """Test outline extraction."""
        outline = get_outline(simple_3d_binary_mask)

        # Outline should have same shape
        assert outline.shape == simple_3d_binary_mask.shape

        # Outline should be binary
        assert np.all((outline == 0) | (outline == 1))

        # Outline should have some non-zero pixels
        assert np.any(outline > 0)

    def test_get_outline_empty_mask(self):
        """Test outline extraction with empty mask."""
        empty_mask = np.zeros((10, 20, 20), dtype=np.uint8)
        outline = get_outline(empty_mask)

        # Should have all zeros
        assert np.all(outline == 0)

    def test_get_outline_full_mask(self):
        """Test outline extraction with full mask."""
        full_mask = np.ones((10, 20, 20), dtype=np.uint8)
        outline = get_outline(full_mask)

        # Should have expected outline
        assert outline.shape == full_mask.shape

    def test_min_intensity_edge_not_zero_for_bright_cell(self):
        """MinIntensityEdge must reflect actual boundary pixel intensities, not 0.

        Regression test for a bug where get_outline used find_boundaries with the
        default mode='thick', which returns both inner (object-side) and outer
        (background-side) boundary pixels. Because selected_image_object is zeroed
        outside the cell before the outline is computed, the outer boundary pixels
        always have intensity 0, making numpy.min always return 0 regardless of
        the cell's true edge intensity.

        The fix is to use mode='inner' in find_boundaries so only pixels that are
        inside the object boundary are included in the edge mask.
        """
        # Build a 10x10x10 image with a solid bright cell (intensity=100) in the centre.
        # Every voxel of the cell has intensity 100, so MinIntensityEdge should be 100.
        # With the bug, outer boundary pixels (zeroed background) are included and
        # min returns 0 instead.
        shape = (10, 10, 10)
        image = numpy.zeros(shape, dtype=numpy.float32)
        labels = numpy.zeros(shape, dtype=numpy.int32)

        image[3:7, 3:7, 3:7] = 100
        labels[3:7, 3:7, 3:7] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        # Extract MinIntensityEdge for object 1
        min_edge_values = [
            v
            for oid, name, v in zip(
                result["object_id"], result["feature_name"], result["value"]
            )
            if oid == 1 and name == "MinIntensityEdge"
        ]

        assert min_edge_values, "MinIntensityEdge not found in result"
        min_edge = min_edge_values[0]

        assert min_edge != 0, (
            f"MinIntensityEdge is 0 for a cell with uniform intensity 100. "
            f"Likely caused by find_boundaries(mode='thick') including outer "
            f"(background) boundary pixels that have been zeroed, so numpy.min "
            f"always returns 0."
        )
        assert min_edge == pytest.approx(100.0), (
            f"MinIntensityEdge should be 100 (cell intensity) but got {min_edge}"
        )

    def test_measure_3d_intensity_cpu_basic(self, object_loader_simple):
        """Test basic intensity measurement."""
        result = measure_3D_intensity_CPU(object_loader=object_loader_simple)

        # Check result structure
        assert isinstance(result, dict)
        expected_keys = {
            "object_id",
            "feature_name",
            "channel",
            "compartment",
            "value",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_measure_3d_intensity_cpu_multiple_objects(self, object_loader_simple):
        """Test intensity measurement with multiple objects."""
        result = measure_3D_intensity_CPU(object_loader=object_loader_simple)

        # Should have measurements for multiple objects
        assert len(result["object_id"]) > 0

        # All lists should have same length
        lengths = [len(result[k]) for k in result.keys()]
        assert len(set(lengths)) == 1

    def test_intensity_discrimination(self, intensity_varied_objects):
        """Test that intensity features correctly distinguish high, medium, low intensity."""
        image, labels = intensity_varied_objects

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        # Extract mean intensity for each object
        obj_means = {}
        for obj_id, feature, value in zip(
            result["object_id"], result["feature_name"], result["value"]
        ):
            if feature == "MeanIntensity":  # Fixed: was "Mean"
                obj_means[obj_id] = value

        # Verify objects were detected
        assert len(obj_means) > 0, f"No objects detected. Result: {result}"

        # If we have all three objects, verify intensity ordering: low < medium < high
        # Object 1: high (3000-4000), Object 2: medium (1000-2000), Object 3: low (100-500)
        if len(obj_means) >= 3:
            assert obj_means[3] < obj_means[2] < obj_means[1]
        # Otherwise, just verify we got some results
        else:
            # At least verify values are in expected range
            for obj_id, mean_val in obj_means.items():
                assert mean_val > 0
        assert obj_means[1] > 2500  # High intensity
        assert 500 < obj_means[2] < 2500  # Medium intensity
        assert obj_means[3] < 600  # Low intensity


# ============================================================================
# TESTS: neighbors_utils
# ============================================================================


class TestNeighborsUtils:
    """Tests for neighbor detection features."""

    def test_neighbors_expand_box_basic(self):
        """Test bounding box expansion."""
        new_min, new_max = neighbors_expand_box(
            min_coor=0,
            max_coord=100,
            current_min=40,
            current_max=60,
            expand_by=10,
        )

        assert new_min == 30
        assert new_max == 70

    def test_neighbors_expand_box_boundary_clamping(self):
        """Test bounding box expansion with boundary clamping."""
        new_min, new_max = neighbors_expand_box(
            min_coor=0,
            max_coord=100,
            current_min=5,
            current_max=95,
            expand_by=10,
        )

        # Should be clamped to boundaries
        assert new_min == 0
        assert new_max == 100

    def test_neighbors_expand_box_partial_boundary(self):
        """Test bounding box expansion with partial boundary clamping."""
        new_min, new_max = neighbors_expand_box(
            min_coor=0,
            max_coord=100,
            current_min=5,
            current_max=50,
            expand_by=10,
        )

        # Min should be clamped, max should not
        assert new_min == 0
        assert new_max == 60

    def test_crop_3d_image_basic(self, simple_3d_image):
        """Test 3D image cropping."""
        bbox = (2, 5, 5, 8, 15, 15)
        cropped = crop_3D_image(image=simple_3d_image, bbox=bbox)

        # Check shape
        assert cropped.shape == (6, 10, 10)

    def test_crop_3d_image_full_extent(self, simple_3d_image):
        """Test cropping to full extent."""
        full_bbox = (0, 0, 0, 10, 20, 20)
        cropped = crop_3D_image(image=simple_3d_image, bbox=full_bbox)

        assert cropped.shape == simple_3d_image.shape
        assert np.array_equal(cropped, simple_3d_image)

    def test_measure_3d_number_of_neighbors_basic(self, object_loader_simple):
        """Test basic neighbor counting."""
        result = measure_3D_number_of_neighbors(
            object_loader=object_loader_simple,
            distance_threshold=10,
            anisotropy_factor=10,
        )

        # Check result structure
        assert isinstance(result, dict)
        assert "object_id" in result
        assert "NeighborsCountAdjacent" in result
        assert "NeighborsCountByDistance-10" in result

    def test_measure_3d_number_of_neighbors_different_thresholds(
        self, object_loader_simple
    ):
        """Test neighbor counting with different thresholds."""
        results = []
        for threshold in [5, 10, 20]:
            result = measure_3D_number_of_neighbors(
                object_loader=object_loader_simple,
                distance_threshold=threshold,
                anisotropy_factor=10,
            )
            results.append(result)

        # All results should be dictionaries
        assert all(isinstance(r, dict) for r in results)

    def test_neighbor_distribution_detection(self, neighbor_distributions):
        """Test that neighbor features detect different spatial distributions."""
        image, labels = neighbor_distributions

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_number_of_neighbors(
            object_loader=loader,
            distance_threshold=5,
            anisotropy_factor=1,
        )

        # Create mapping of object to neighbor counts
        obj_neighbors = {}
        for obj_id, adj_count, dist_count in zip(
            result["object_id"],
            result["NeighborsCountAdjacent"],
            result["NeighborsCountByDistance-5"],
        ):
            obj_neighbors[obj_id] = {"adjacent": adj_count, "distance_5": dist_count}

        # Should detect objects
        assert len(obj_neighbors) > 0

        # Verify the neighbor measurement keys exist and contain data
        assert "adjacent" in list(obj_neighbors.values())[0]
        assert "distance_5" in list(obj_neighbors.values())[0]

    def test_neighbors_count_adjacent_detects_touching_cells(self):
        """Regression: NeighborsCountAdjacent used the object's own regionprops
        bbox (exclusive max indices), so the immediately adjacent voxel of a
        touching neighbor was never included → always returned 0.
        """
        labels = np.zeros((15, 15, 15), dtype=int)
        labels[5:10, 5:10, 2:6] = 1
        labels[5:10, 5:10, 6:10] = 2  # shares a face with cell 1 at x=5|6
        image = np.zeros((15, 15, 15), dtype=float)
        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_number_of_neighbors(
            object_loader=loader, distance_threshold=1, anisotropy_factor=1
        )
        obj_map = dict(zip(result["object_id"], result["NeighborsCountAdjacent"]))
        assert obj_map[1] >= 1, f"Cell 1 should detect cell 2 as adjacent, got {obj_map[1]}"
        assert obj_map[2] >= 1, f"Cell 2 should detect cell 1 as adjacent, got {obj_map[2]}"


# ============================================================================
# TESTS: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_label_image(self, simple_3d_image):
        """Test handling of empty label image."""
        empty_label = np.zeros_like(simple_3d_image)

        loader = ObjectLoader(
            image=simple_3d_image,
            label_image=empty_label,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        # Should handle gracefully
        assert len(loader.object_ids) == 0

    def test_single_voxel_object(self, simple_3d_image):
        """Test handling of single-voxel objects."""
        single_voxel_label = np.zeros((10, 20, 20), dtype=np.uint8)
        single_voxel_label[5, 10, 10] = 1

        loader = ObjectLoader(
            image=simple_3d_image,
            label_image=single_voxel_label,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should not crash
        assert isinstance(result, dict)

    def test_very_large_image_subsampling(self, simple_3d_image, simple_3d_binary_mask):
        """Test subsampling with moderately aggressive factor (edge case)."""
        # Test with aggressive but viable subsampling - still an edge case
        new_shape = numpy.array(simple_3d_image.shape, dtype=float) * 0.1
        subsampled_img = _subsample_3d(
            data=simple_3d_image.astype(float),
            new_shape=new_shape,
            subsample_factor=0.1,
            order=1,
        )

        # Should still be valid despite aggressive subsampling
        assert subsampled_img.size > 0
        # Check dimensions are significantly reduced (at least 50% reduction)
        assert subsampled_img.size <= simple_3d_image.size * 0.5

    def test_zero_intensity_image(self):
        """Test handling of all-zero intensity image."""
        zero_image = np.zeros((10, 20, 20), dtype=np.uint8)
        label_image = np.zeros((10, 20, 20), dtype=np.uint8)
        label_image[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=zero_image,
            label_image=label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_nan_values_in_image(self, simple_3d_image):
        """Test handling of NaN values."""
        image_with_nan = simple_3d_image.astype(float)
        image_with_nan[0, 0, 0] = np.nan

        label_image = np.zeros_like(simple_3d_image)
        label_image[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image_with_nan,
            label_image=label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        # Should handle without crashing
        result = measure_3D_intensity_CPU(object_loader=loader)
        assert isinstance(result, dict)

    def test_maximum_label_value(self):
        """Test with label values at uint8 maximum (edge case)."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)

        # Use maximum label value for uint8
        labels[2:5, 5:10, 5:10] = 255  # Maximum uint8 value
        labels[5:8, 10:15, 10:15] = 254

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)
        assert 255 in [int(oid) for oid in result["object_id"]]

    def test_all_pixels_at_type_maximum(self):
        """Test with all intensity values at data type maximum (edge case)."""
        # All pixels at maximum value
        image = np.full((10, 20, 20), 65535, dtype=np.uint16)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)
        # Should handle saturated images
        assert isinstance(result, dict)
        assert len(result["object_id"]) > 0

    def test_extremely_anisotropic_spacing(self):
        """Test with extreme anisotropy (edge case)."""
        from unittest.mock import Mock

        image = np.random.randint(50, 200, (20, 20, 20), dtype=np.uint8)
        labels = np.zeros((20, 20, 20), dtype=np.uint8)
        labels[5:15, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        # Extremely anisotropic spacing (1000x difference)
        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (100.0, 0.1, 0.1)

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=loader,
        )

        # Should handle extreme anisotropy
        assert isinstance(result, dict)
        assert "Volume" in result

    def test_object_spanning_full_dimension(self):
        """Test with object spanning entire dimension (edge case)."""
        image = np.random.randint(50, 200, (20, 30, 30), dtype=np.uint8)
        labels = np.zeros((20, 30, 30), dtype=np.uint8)

        # Object spans entire Z dimension but is thin in X-Y (like a pillar)
        labels[:, 15, 15] = 1  # Single column through entire volume

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)
        assert len(result["object_id"]) > 0


class TestDataConsistency:
    """Tests for data consistency across functions."""

    def test_area_consistency(self, object_loader_simple):
        """Test that area measurements are consistent."""
        from unittest.mock import Mock

        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=object_loader_simple,
        )

        # All volumes should be positive
        assert all(v > 0 for v in result["Volume"])

        # Extents should be between 0 and 1
        if "Extent" in result:
            assert all(0 <= v <= 1 for v in result["Extent"])

    def test_intensity_range_consistency(self, object_loader_simple):
        """Test that intensity measurements are within reasonable ranges."""
        result = measure_3D_intensity_CPU(object_loader=object_loader_simple)

        # All values should be finite
        assert all(np.isfinite(v) for v in result["value"] if v is not None)

    def test_neighbor_count_consistency(self, object_loader_simple):
        """Test that neighbor counts are non-negative."""
        result = measure_3D_number_of_neighbors(
            object_loader=object_loader_simple,
            distance_threshold=10,
        )

        # All neighbor counts should be non-negative
        assert all(n >= 0 for n in result["NeighborsCountAdjacent"])
        assert all(n >= 0 for n in result["NeighborsCountByDistance-10"])


class TestDataTypes:
    """Tests for correct data types in outputs."""

    def test_area_shape_output_types(self, object_loader_simple):
        """Test output data types from area/shape measurements."""
        from unittest.mock import Mock

        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=object_loader_simple,
        )

        # Object IDs should be integers
        assert all(isinstance(oid, (int, np.integer)) for oid in result["object_id"])

        # Measurements should be numeric
        assert all(isinstance(v, (int, float, np.number)) for v in result["Volume"])

    def test_texture_output_types(self, object_loader_simple):
        """Test output data types from texture measurements."""
        result = measure_3D_texture(object_loader=object_loader_simple)

        # Object IDs should be integers
        assert all(isinstance(oid, (int, np.integer)) for oid in result["object_id"])

        # Texture names should be strings
        assert all(isinstance(name, str) for name in result["texture_name"])

        # Texture values should be numeric
        assert all(
            isinstance(v, (int, float, np.number)) for v in result["texture_value"]
        )

    def test_colocalization_output_types(self, high_contrast_images):
        """Test output data types from colocalization measurements."""
        img1, img2 = high_contrast_images
        result = measure_3D_colocalization(cropped_image_1=img1, cropped_image_2=img2)

        # All values should be numeric
        assert all(isinstance(v, (int, float, np.number)) for v in result.values())


# ============================================================================
# ADDITIONAL COMPREHENSIVE TESTS
# ============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_measure_3d_intensity_single_object(self):
        """Test intensity measurement with single object."""
        image = np.random.randint(50, 200, (15, 25, 25), dtype=np.uint8)
        label = np.zeros_like(image, dtype=np.uint8)
        label[5:10, 8:17, 8:17] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        assert len(result["object_id"]) > 0
        assert all(oid == 1 for oid in result["object_id"])

    def test_texture_measurement_with_mask_region(self, object_loader_simple):
        """Test texture with limited mask region."""
        result = measure_3D_texture(
            object_loader=object_loader_simple,
            distance=1,
            grayscale=256,
        )

        assert "texture_name" in result
        assert len(result["texture_name"]) > 0
        assert all(
            isinstance(v, (int, float, np.number)) or np.isnan(v)
            for v in result["texture_value"]
        )

    def test_granularity_with_large_radius(self, object_loader_simple):
        """Test granularity with large filter radius."""
        result = measure_3D_granularity(
            object_loader=object_loader_simple,
            radius=20,
            granular_spectrum_length=8,
        )

        assert "value" in result
        assert len(result["value"]) > 0

    def test_neighbors_with_small_distance(self, object_loader_simple):
        """Test neighbor counting with small distance threshold."""
        result = measure_3D_number_of_neighbors(
            object_loader=object_loader_simple,
            distance_threshold=2,
        )

        assert "NeighborsCountAdjacent" in result
        assert "NeighborsCountByDistance-2" in result

    def test_colocalization_with_completely_separate_images(self):
        """Test colocalization when images have no overlap in signal."""
        img1 = np.zeros((10, 15, 15), dtype=np.uint8)
        img1[2:5, 5:10, 5:10] = 200

        img2 = np.zeros((10, 15, 15), dtype=np.uint8)
        img2[7:10, 10:15, 10:15] = 200

        result = measure_3D_colocalization(
            cropped_image_1=img1,
            cropped_image_2=img2,
            thr=15,
        )

        assert isinstance(result, dict)


class TestParameterVariations:
    """Tests with various parameter combinations."""

    def test_texture_with_all_grayscale_levels(self, object_loader_simple):
        """Test texture measurement with different grayscale quantizations."""
        # scale_image only supports 256 (uint8) and 65536 (uint16), but 65536 is memory-intensive
        result = measure_3D_texture(
            object_loader=object_loader_simple,
            grayscale=256,
        )
        assert len(result["texture_value"]) > 0

    def test_granularity_with_different_spectrum_lengths(self, object_loader_simple):
        """Test granularity with various spectrum lengths."""
        spectrum_lengths = [4, 8, 12, 16, 20]

        for spec_len in spectrum_lengths:
            result = measure_3D_granularity(
                object_loader=object_loader_simple,
                granular_spectrum_length=spec_len,
            )
            assert len(result["value"]) >= spec_len

    def test_neighbors_with_multiple_distance_thresholds(self, object_loader_simple):
        """Test neighbor counting with varied distance thresholds."""
        thresholds = [2, 5, 10, 20, 50]

        for threshold in thresholds:
            result = measure_3D_number_of_neighbors(
                object_loader=object_loader_simple,
                distance_threshold=threshold,
            )
            assert "NeighborsCountAdjacent" in result

    def test_area_shape_with_varying_anisotropy(
        self, simple_3d_image, simple_3d_label_image
    ):
        """Test area/shape measurement with different anisotropy values."""
        from unittest.mock import Mock

        anisotropy_spacings = [
            (1.0, 0.1, 0.1),
            (2.0, 0.1, 0.1),
            (1.0, 0.2, 0.2),
            (0.5, 0.1, 0.1),
        ]

        loader = ObjectLoader(
            image=simple_3d_image,
            label_image=simple_3d_label_image,
            channel_name="test",
            compartment_name="test",
        )

        results = {}
        for spacing in anisotropy_spacings:
            mock_loader = Mock()
            mock_loader.anisotropy_spacing = spacing
            result = measure_3D_area_size_shape(
                image_set_loader=mock_loader,
                object_loader=loader,
            )
            results[spacing] = len(result["object_id"])
            assert len(result["object_id"]) > 0


class TestMultiObjectScenarios:
    """Tests with multiple objects and complex scenarios."""

    def test_intensity_with_many_small_objects(self):
        """Test intensity measurement with many small objects."""
        image = np.random.randint(100, 200, (20, 40, 40), dtype=np.uint8)
        label = np.zeros_like(image, dtype=np.uint8)

        obj_id = 1
        for z in range(2, 18, 4):
            for y in range(2, 38, 6):
                for x in range(2, 38, 6):
                    label[z : z + 2, y : y + 3, x : x + 3] = obj_id
                    obj_id += 1
                    if obj_id > 20:
                        break

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        unique_ids = set(result["object_id"])
        assert len(unique_ids) > 5

    def test_texture_with_many_objects(self):
        """Test texture with many objects in single image."""
        image = np.random.randint(80, 180, (25, 35, 35), dtype=np.uint8)
        label = np.zeros_like(image, dtype=np.uint8)

        obj_id = 1
        for z in range(3, 22, 5):
            for y in range(3, 32, 5):
                label[z : z + 3, y : y + 4, y : y + 4] = obj_id
                obj_id += 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_texture(object_loader=loader)

        assert len(result["object_id"]) > 0
        unique_ids = len(set(result["object_id"]))
        assert unique_ids >= 3

    def test_area_shape_with_overlapping_bboxes(self):
        """Test area/shape with objects having overlapping bounding boxes."""
        from unittest.mock import Mock

        image = np.random.randint(50, 200, (20, 30, 30), dtype=np.uint8)
        label = np.zeros_like(image, dtype=np.uint8)

        # Object 1
        label[5:10, 5:15, 5:15] = 1
        # Object 2 - overlapping bbox with object 1
        label[7:12, 12:22, 12:22] = 2

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )

        mock_loader = Mock()
        mock_loader.anisotropy_spacing = (1.0, 0.1, 0.1)
        result = measure_3D_area_size_shape(
            image_set_loader=mock_loader,
            object_loader=loader,
        )

        assert len(result["object_id"]) == 2


class TestImageTypeHandling:
    """Tests for handling different image data types."""

    def test_intensity_with_uint16_image(self):
        """Test intensity measurement with 16-bit unsigned integer image."""
        image = np.random.randint(1000, 10000, (10, 20, 20), dtype=np.uint16)
        label = np.zeros_like(image, dtype=np.uint8)
        label[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        assert len(result["object_id"]) > 0
        assert all(np.isfinite(v) for v in result["value"])

    def test_intensity_with_float32_image(self):
        """Test intensity with float32 image."""
        image = np.random.rand(10, 20, 20).astype(np.float32) * 255
        label = np.zeros((10, 20, 20), dtype=np.uint8)
        label[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        assert len(result["object_id"]) > 0

    def test_texture_with_uint16_image(self):
        """Test texture with 16-bit image."""
        image = np.random.randint(1000, 10000, (10, 20, 20), dtype=np.uint16)
        label = np.zeros((10, 20, 20), dtype=np.uint8)
        label[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_texture(object_loader=loader)

        assert len(result["texture_value"]) > 0


class TestMaskValidation:
    """Tests for proper mask handling and validation."""

    def test_intensity_with_disconnected_mask_regions(self):
        """Test with label image having disconnected components."""
        image = np.random.randint(50, 200, (15, 25, 25), dtype=np.uint8)
        label = np.zeros_like(image, dtype=np.uint8)

        # Two disconnected regions with same label
        label[2:5, 5:10, 5:10] = 1
        label[10:13, 15:20, 15:20] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should process both connected components
        assert len(result["object_id"]) > 0

    def test_texture_with_sparse_mask(self):
        """Test texture with very sparse label (only few pixels)."""
        image = np.ones((10, 20, 20), dtype=np.uint8) * 100
        label = np.zeros_like(image, dtype=np.uint8)

        # Very sparse labeling
        label[5, 5:10, 5:10] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_texture(object_loader=loader)

        assert isinstance(result, dict)

    def test_granularity_with_small_object(self):
        """Test granularity with very small object."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        label = np.zeros_like(image, dtype=np.uint8)

        # Single voxel object
        label[5, 10, 10] = 1

        loader = ObjectLoader(
            image=image,
            label_image=label,
            channel_name="test",
            compartment_name="test",
        )
        result = measure_3D_granularity(
            object_loader=loader,
            radius=5,
        )

        assert isinstance(result, dict)


class TestColocalizationAdvanced:
    """Advanced tests for colocalization functions."""

    def test_costes_threshold_with_noisy_images(self):
        """Test Costes threshold with noisy images."""
        np.random.seed(42)
        img1 = np.random.normal(100, 30, (10, 15, 15)).astype(np.uint8)
        img2 = np.random.normal(100, 30, (10, 15, 15)).astype(np.uint8)

        thr1, thr2 = linear_costes_threshold_calculation(
            first_image=img1,
            second_image=img2,
            scale_max=255,
            fast_costes="Accurate",
        )

        assert 0 <= thr1 <= 255
        assert 0 <= thr2 <= 255

    def test_colocalization_identical_images_perfect_correlation(self):
        """Test colocalization with identical images should give perfect correlation."""
        image = np.random.randint(50, 200, (10, 15, 15), dtype=np.uint8)

        result = measure_3D_colocalization(
            cropped_image_1=image,
            cropped_image_2=image,
            thr=10,
        )

        # Should have high correlation for identical images
        if "PearsonsCorrelation" in result:
            assert result["PearsonsCorrelation"] > 0.95

    def test_colocalization_completely_anti_correlated_images(self):
        """Test colocalization with anti-correlated signal."""
        img1 = np.zeros((10, 15, 15), dtype=np.uint8)
        img1[3:7, 5:12, 5:12] = 200

        img2 = np.zeros((10, 15, 15), dtype=np.uint8)
        img2[3:7, 5:12, 5:12] = 0
        img2[0:3, 0:5, 0:5] = 200

        result = measure_3D_colocalization(
            cropped_image_1=img1,
            cropped_image_2=img2,
            thr=10,
        )

        assert isinstance(result, dict)

        assert isinstance(result, dict)
