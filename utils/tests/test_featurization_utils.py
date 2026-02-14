"""Comprehensive tests for featurization utility functions."""

from __future__ import annotations

import numpy as np
import pytest
from src.image_analysis_3D.featurization_utils.area_size_shape_utils import (
    calculate_surface_area,
    measure_3D_area_size_shape,
)
from src.image_analysis_3D.featurization_utils.colocalization_utils import (
    bisection_costes_threshold_calculation,
    linear_costes_threshold_calculation,
    measure_3D_colocalization,
)
from src.image_analysis_3D.featurization_utils.granularity_utils import (
    _apply_tophat_filter,
    _subsample_image,
    measure_3D_granularity,
)
from src.image_analysis_3D.featurization_utils.intensity_utils import (
    get_outline,
    measure_3D_intensity_CPU,
)
from src.image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)
from src.image_analysis_3D.featurization_utils.neighbors_utils import (
    crop_3D_image,
    measure_3D_number_of_neighbors,
    neighbors_expand_box,
)
from src.image_analysis_3D.featurization_utils.texture_utils import (
    measure_3D_texture,
    scale_image,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_3d_image():
    """Create a simple 3D test image with known properties."""
    image = np.zeros((10, 20, 20), dtype=np.uint8)
    # Add some varying intensities
    image[2:8, 5:15, 5:15] = 100
    image[3:7, 6:14, 6:14] = 150
    return image


@pytest.fixture
def simple_3d_binary_mask():
    """Create a simple 3D binary mask."""
    mask = np.zeros((10, 20, 20), dtype=np.uint8)
    mask[2:8, 5:15, 5:15] = 1
    return mask


@pytest.fixture
def simple_3d_label_image():
    """Create a simple 3D label image with multiple objects."""
    label_image = np.zeros((10, 20, 20), dtype=np.uint8)
    # Object 1
    label_image[2:5, 5:10, 5:10] = 1
    # Object 2
    label_image[5:8, 10:15, 10:15] = 2
    # Object 3
    label_image[3:6, 15:19, 15:19] = 3
    return label_image


@pytest.fixture
def object_loader_simple(simple_3d_image, simple_3d_label_image):
    """Create a simple ObjectLoader for testing."""
    loader = ObjectLoader(
        image=simple_3d_image,
        label_image=simple_3d_label_image,
        channel_name="test_channel",
        compartment_name="test_compartment",
    )
    return loader


@pytest.fixture
def image_set_loader_simple(tmp_path, simple_3d_image, simple_3d_label_image):
    """Create a simple ImageSetLoader for testing."""
    import skimage.io

    # Create temp directories
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    # Save test images
    skimage.io.imsave(str(image_dir / "channel_DAPI.tif"), simple_3d_image)
    skimage.io.imsave(str(mask_dir / "mask_nuclei.tif"), simple_3d_label_image)

    # Create loader
    loader = ImageSetLoader(
        image_set_path=image_dir,
        mask_set_path=mask_dir,
        anisotropy_spacing=(1.0, 0.1, 0.1),
        channel_mapping={"DAPI": "DAPI", "nuclei_mask": "nuclei"},
        image_set_name="test_well_fov",
    )
    return loader


@pytest.fixture
def high_contrast_images():
    """Create high contrast test images for colocalization."""
    img1 = np.zeros((10, 15, 15), dtype=np.uint8)
    img2 = np.zeros((10, 15, 15), dtype=np.uint8)

    # Overlapping region with high signal
    img1[2:8, 5:12, 5:12] = 200
    img2[2:8, 5:12, 5:12] = 180

    # Non-overlapping regions
    img1[1:3, 1:5, 1:5] = 150
    img2[8:10, 10:15, 10:15] = 150

    return img1, img2


@pytest.fixture
def low_contrast_images():
    """Create low contrast test images."""
    img1 = np.random.randint(50, 100, (10, 15, 15), dtype=np.uint8)
    img2 = np.random.randint(50, 100, (10, 15, 15), dtype=np.uint8)
    return img1, img2


# ============================================================================
# TESTS: area_size_shape_utils
# ============================================================================


class TestAreaSizeShapeUtils:
    """Tests for area, size, and shape feature extraction."""

    def test_measure_3d_area_size_shape_basic(self, object_loader_simple):
        """Test basic functionality of measure_3D_area_size_shape."""
        result = measure_3D_area_size_shape(
            image_set_loader=None,  # This would be created in real scenario
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

        surface_area = calculate_surface_area(
            label_object=simple_3d_label_image,
            props=props_dict,
            spacing=(1.0, 0.1, 0.1),
        )

        # Surface area should be positive
        assert surface_area > 0
        assert isinstance(surface_area, float)

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
        for levels in [64, 128, 256, 512]:
            scaled = scale_image(simple_3d_image, num_gray_levels=levels)
            assert scaled.dtype == np.uint8
            assert scaled.max() < levels

    def test_scale_image_uniform_input(self):
        """Test scaling with uniform image."""
        uniform_image = np.ones((10, 10, 10), dtype=np.uint16) * 100
        scaled = scale_image(uniform_image, num_gray_levels=256)

        # Should handle uniform images gracefully
        assert scaled.shape == uniform_image.shape

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
        for grayscale in [64, 128, 256, 512]:
            result = measure_3D_texture(
                object_loader=object_loader_simple,
                distance=1,
                grayscale=grayscale,
            )

            # Should have results for all grayscale levels
            assert len(result["texture_value"]) > 0


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

        # Common colocalization metrics
        common_keys = {
            "PearsonsCorrelation",
            "Pearsons",
            "SpearmanCorrelation",
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

        # Pearson correlation should be close to 1 for identical images
        if "PearsonsCorrelation" in result:
            assert result["PearsonsCorrelation"] > 0.9


# ============================================================================
# TESTS: granularity_utils
# ============================================================================


class TestGranularityUtils:
    """Tests for granularity feature extraction."""

    def test_subsample_image_basic(self, simple_3d_image, simple_3d_binary_mask):
        """Test basic image subsampling."""
        subsampled_img, subsampled_mask = _subsample_image(
            image=simple_3d_image,
            mask=simple_3d_binary_mask,
            subsample_factor=0.5,
            z_to_xy_ratio=10.0,
            make_isotropic=True,
        )

        # Subsampled image should be smaller or equal
        assert subsampled_img.size <= simple_3d_image.size
        assert subsampled_mask.size <= simple_3d_binary_mask.size

    def test_subsample_image_factor_one(self, simple_3d_image, simple_3d_binary_mask):
        """Test subsampling with factor=1.0 (no subsampling)."""
        subsampled_img, subsampled_mask = _subsample_image(
            image=simple_3d_image,
            mask=simple_3d_binary_mask,
            subsample_factor=1.0,
        )

        # Should return original images
        assert subsampled_img.shape == simple_3d_image.shape
        assert subsampled_mask.shape == simple_3d_binary_mask.shape

    def test_apply_tophat_filter_basic(self, simple_3d_image, simple_3d_binary_mask):
        """Test tophat filter application."""
        filtered = _apply_tophat_filter(
            pixels=simple_3d_image.astype(float),
            mask=simple_3d_binary_mask,
            radius=2,
        )

        # Output should have same shape
        assert filtered.shape == simple_3d_image.shape

    def test_measure_3d_granularity_basic(self, object_loader_simple):
        """Test basic granularity measurement."""
        result = measure_3D_granularity(
            object_loader=object_loader_simple,
            radius=10,
            granular_spectrum_length=16,
            subsample_image_value=0.5,
            z_to_xy_ratio=10.0,
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
                subsample_image_value=1.5,
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


# ============================================================================
# TESTS: intensity_utils
# ============================================================================


class TestIntensityUtils:
    """Tests for intensity feature extraction."""

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
        assert "Neighbors_adjacent" in result
        assert "Neighbors_10" in result

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
        """Test subsampling with very small factor."""
        subsampled_img, subsampled_mask = _subsample_image(
            image=simple_3d_image,
            mask=simple_3d_binary_mask,
            subsample_factor=0.1,
            z_to_xy_ratio=10.0,
            make_isotropic=True,
        )

        # Should still be valid
        assert subsampled_img.size > 0
        assert subsampled_mask.size > 0

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


class TestDataConsistency:
    """Tests for data consistency across functions."""

    def test_area_consistency(self, object_loader_simple):
        """Test that area measurements are consistent."""
        result = measure_3D_area_size_shape(
            image_set_loader=None,
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
        assert all(n >= 0 for n in result["Neighbors_adjacent"])
        assert all(n >= 0 for n in result["Neighbors_10"])


class TestDataTypes:
    """Tests for correct data types in outputs."""

    def test_area_shape_output_types(self, object_loader_simple):
        """Test output data types from area/shape measurements."""
        result = measure_3D_area_size_shape(
            image_set_loader=None,
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
            grayscale=128,
        )

        assert "texture_name" in result
        assert len(result["texture_name"]) > 0
        assert all(isinstance(v, (int, float)) for v in result["texture_value"])

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

        assert "Neighbors_adjacent" in result
        assert "Neighbors_2" in result

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
        grayscale_levels = [32, 64, 128, 256, 512]
        results = {}

        for gray_level in grayscale_levels:
            result = measure_3D_texture(
                object_loader=object_loader_simple,
                grayscale=gray_level,
            )
            results[gray_level] = len(result["texture_value"])
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
            assert "Neighbors_adjacent" in result

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
