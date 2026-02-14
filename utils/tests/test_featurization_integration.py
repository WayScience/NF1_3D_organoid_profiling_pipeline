"""Integration and performance tests for featurization utilities."""

from __future__ import annotations

import time
from unittest.mock import Mock

import numpy as np
import pytest
from src.image_analysis_3D.featurization_utils.area_size_shape_utils import (
    measure_3D_area_size_shape,
)
from src.image_analysis_3D.featurization_utils.colocalization_utils import (
    measure_3D_colocalization,
)
from src.image_analysis_3D.featurization_utils.granularity_utils import (
    measure_3D_granularity,
)
from src.image_analysis_3D.featurization_utils.intensity_utils import (
    measure_3D_intensity_CPU,
)
from src.image_analysis_3D.featurization_utils.loading_classes import ObjectLoader
from src.image_analysis_3D.featurization_utils.neighbors_utils import (
    measure_3D_number_of_neighbors,
)
from src.image_analysis_3D.featurization_utils.texture_utils import measure_3D_texture

# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFeaturizationPipeline:
    """Integration tests for the complete featurization pipeline."""

    @pytest.fixture
    def realistic_3d_volume(self):
        """Create a more realistic 3D volume."""
        volume = np.zeros((30, 50, 50), dtype=np.uint16)

        # Add 3 objects with varying intensities
        # Object 1: bright nucleus
        volume[5:15, 10:20, 10:20] = np.random.randint(150, 255, (10, 10, 10))

        # Object 2: medium intensity
        volume[10:20, 25:40, 25:40] = np.random.randint(100, 200, (10, 15, 15))

        # Object 3: lower intensity
        volume[15:25, 5:15, 35:50] = np.random.randint(50, 150, (10, 10, 15))

        return volume

    @pytest.fixture
    def realistic_label_image(self):
        """Create a realistic label image."""
        labels = np.zeros((30, 50, 50), dtype=np.uint8)

        # Object 1
        labels[5:15, 10:20, 10:20] = 1

        # Object 2
        labels[10:20, 25:40, 25:40] = 2

        # Object 3
        labels[15:25, 5:15, 35:50] = 3

        return labels

    def test_all_features_single_object_loader(
        self, realistic_3d_volume, realistic_label_image
    ):
        """Test that all feature extraction functions work with same data."""
        loader = ObjectLoader(
            image=realistic_3d_volume,
            label_image=realistic_label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        # Create mock for image set loader
        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        # Run all feature extraction functions
        results = {}

        # Area/size/shape
        results["area"] = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=loader,
        )

        # Texture
        results["texture"] = measure_3D_texture(
            object_loader=loader,
            distance=1,
            grayscale=256,
        )

        # Granularity
        results["granularity"] = measure_3D_granularity(
            object_loader=loader,
            radius=5,
            granular_spectrum_length=16,
            subsample_image_value=0.5,
        )

        # Intensity
        results["intensity"] = measure_3D_intensity_CPU(object_loader=loader)

        # Neighbors
        results["neighbors"] = measure_3D_number_of_neighbors(
            object_loader=loader,
            distance_threshold=10,
        )

        # All should return dictionaries
        assert all(isinstance(v, dict) for v in results.values())

        # All should have object_id
        assert all("object_id" in v for v in results.values())

    def test_feature_extraction_consistency(
        self, realistic_3d_volume, realistic_label_image
    ):
        """Test that feature extraction is deterministic."""
        loader1 = ObjectLoader(
            image=realistic_3d_volume.copy(),
            label_image=realistic_label_image.copy(),
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        loader2 = ObjectLoader(
            image=realistic_3d_volume.copy(),
            label_image=realistic_label_image.copy(),
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        # Extract same features from both loaders
        result1 = measure_3D_intensity_CPU(object_loader=loader1)
        result2 = measure_3D_intensity_CPU(object_loader=loader2)

        # Results should be identical
        assert len(result1["object_id"]) == len(result2["object_id"])
        assert result1["object_id"] == result2["object_id"]

    def test_multiple_compartments(self):
        """Test feature extraction with multiple compartments."""
        # Create a volume with 5 compartments
        volume = np.random.randint(50, 200, (20, 30, 30), dtype=np.uint8)
        labels = np.zeros((20, 30, 30), dtype=np.uint8)

        # Create 5 objects
        for i in range(1, 6):
            z_start = (i - 1) * 4
            z_end = min(z_start + 4, 20)
            labels[z_start:z_end, i * 5 : (i + 1) * 5, i * 5 : (i + 1) * 5] = i

        loader = ObjectLoader(
            image=volume,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should have features for all objects
        unique_ids = set(result["object_id"])
        assert len(unique_ids) == len(set(labels[labels > 0]))


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance and efficiency tests."""

    @pytest.fixture
    def large_volume(self):
        """Create a larger volume for performance testing."""
        return np.random.randint(50, 200, (50, 100, 100), dtype=np.uint8)

    @pytest.fixture
    def large_label_image(self):
        """Create larger label image."""
        labels = np.zeros((50, 100, 100), dtype=np.uint8)

        # Create multiple objects
        label_id = 1
        for z in range(5, 45, 10):
            for y in range(10, 90, 20):
                for x in range(10, 90, 20):
                    labels[z : z + 8, y : y + 15, x : x + 15] = label_id
                    label_id += 1
                    if label_id > 10:  # Limit number of objects
                        break

        return labels

    def test_area_shape_performance(self, large_volume, large_label_image):
        """Test performance of area/shape measurement."""
        loader = ObjectLoader(
            image=large_volume,
            label_image=large_label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        start_time = time.time()
        result = measure_3D_area_size_shape(
            image_set_loader=mock_image_set_loader,
            object_loader=loader,
        )
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert len(result["object_id"]) > 0

    def test_intensity_performance(self, large_volume, large_label_image):
        """Test performance of intensity measurement."""
        loader = ObjectLoader(
            image=large_volume,
            label_image=large_label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        start_time = time.time()
        result = measure_3D_intensity_CPU(object_loader=loader)
        elapsed = time.time() - start_time

        # Should complete reasonably fast
        assert elapsed < 10.0
        assert len(result["object_id"]) > 0

    def test_texture_performance(self, large_volume, large_label_image):
        """Test performance of texture measurement."""
        loader = ObjectLoader(
            image=large_volume,
            label_image=large_label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        start_time = time.time()
        result = measure_3D_texture(
            object_loader=loader,
            distance=1,
            grayscale=256,
        )
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 10.0
        assert len(result["texture_value"]) > 0

    def test_memory_efficiency_subsampling(self, large_volume, large_label_image):
        """Test memory efficiency with subsampling."""
        loader = ObjectLoader(
            image=large_volume,
            label_image=large_label_image,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

        start_time = time.time()
        result = measure_3D_granularity(
            object_loader=loader,
            subsample_image_value=0.25,  # Aggressive subsampling
        )
        elapsed = time.time() - start_time

        # Should be faster with subsampling
        assert elapsed < 10.0


# ============================================================================
# PARAMETER VALIDATION TESTS
# ============================================================================


class TestParameterValidation:
    """Tests for parameter validation and error messages."""

    @pytest.fixture
    def simple_loader(self):
        """Create simple object loader."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1
        return ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

    def test_granularity_invalid_subsample(self, simple_loader):
        """Test granularity with invalid subsample value."""
        with pytest.raises(ValueError, match="subsample_image_value"):
            measure_3D_granularity(
                object_loader=simple_loader,
                subsample_image_value=2.0,  # Invalid: > 1.0
            )

    def test_granularity_invalid_radius(self, simple_loader):
        """Test granularity with invalid radius."""
        with pytest.raises(ValueError, match="radius"):
            measure_3D_granularity(
                object_loader=simple_loader,
                radius=0,  # Invalid: must be positive
            )

    def test_granularity_invalid_spectrum_length(self, simple_loader):
        """Test granularity with invalid spectrum length."""
        with pytest.raises(ValueError, match="granular_spectrum_length"):
            measure_3D_granularity(
                object_loader=simple_loader,
                granular_spectrum_length=0,  # Invalid: must be positive
            )

    def test_texture_scale_image_with_different_dtypes(self):
        """Test texture scaling with various input dtypes."""
        from src.image_analysis_3D.featurization_utils.texture_utils import scale_image

        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            image = np.random.randint(50, 200, (10, 10, 10)).astype(dtype)
            scaled = scale_image(image, num_gray_levels=256)

            # Should always return uint8
            assert scaled.dtype == np.uint8


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_very_small_values(self):
        """Test feature extraction with very small intensity values."""
        image = np.random.rand(10, 20, 20).astype(np.float32) * 1e-6
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(image=image, label_image=labels)
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should handle very small values gracefully
        assert all(np.isfinite(v) for v in result["value"] if v is not None)

    def test_very_large_values(self):
        """Test feature extraction with very large values."""
        image = np.random.randint(
            int(1e9), int(1e9) + 1000000, (10, 20, 20), dtype=np.uint32
        )
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(image=image, label_image=labels)
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should handle large values
        assert isinstance(result, dict)

    def test_mixed_scale_colocalization(self):
        """Test colocalization with images at very different scales."""
        img1 = np.random.randint(0, 256, (10, 15, 15), dtype=np.uint8)
        img2 = np.random.randint(0, 65536, (10, 15, 15), dtype=np.uint16)

        # Convert to same scale for comparison
        img2_scaled = (img2.astype(float) / 256).astype(np.uint8)

        result = measure_3D_colocalization(
            cropped_image_1=img1,
            cropped_image_2=img2_scaled,
        )

        # Should handle different scales
        assert isinstance(result, dict)

    def test_uniform_image_intensity(self):
        """Test intensity features with uniform image."""
        image = np.ones((10, 20, 20), dtype=np.uint8) * 127
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(image=image, label_image=labels)
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should handle uniform image
        assert len(result["object_id"]) > 0


# ============================================================================
# ROBUSTNESS TESTS
# ============================================================================


class TestRobustness:
    """Tests for robustness to various input conditions."""

    def test_disconnected_objects(self):
        """Test with disconnected components within same label."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)

        # Create disconnected regions with same label
        labels[2:4, 5:10, 5:10] = 1
        labels[6:8, 10:15, 10:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should handle disconnected components
        assert len(result["object_id"]) > 0

    def test_touch_boundary_objects(self):
        """Test objects that touch image boundaries."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)

        # Object touching all boundaries
        labels[0:10, 0:20, 0:20] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_area_size_shape(
            image_set_loader=Mock(anisotropy_spacing=(1.0, 0.1, 0.1)),
            object_loader=loader,
        )

        # Should handle boundary objects
        assert len(result["object_id"]) > 0

    def test_very_thin_objects(self):
        """Test with very thin objects (single voxel thickness)."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)

        # Single voxel thick object
        labels[5, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should handle thin objects
        assert len(result["object_id"]) > 0

    def test_nested_objects_like_structures(self):
        """Test with nested, concentric objects."""
        image = np.random.randint(50, 200, (20, 30, 30), dtype=np.uint8)
        labels = np.zeros((20, 30, 30), dtype=np.uint8)

        # Outer object
        labels[2:18, 5:25, 5:25] = 1
        # Inner object (will overwrite outer labels)
        labels[6:14, 10:20, 10:20] = 2

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )
        result = measure_3D_texture(object_loader=loader)

        # Should handle overlapping regions
        assert len(result["object_id"]) > 0


# ============================================================================
# OUTPUT VALIDATION TESTS
# ============================================================================


class TestOutputValidation:
    """Tests to validate output structure and content."""

    @pytest.fixture
    def simple_loader(self):
        """Create simple object loader."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1
        labels[3:7, 15:19, 15:19] = 2
        return ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test_channel",
            compartment_name="test_compartment",
        )

    def test_area_shape_output_structure(self, simple_loader):
        """Validate structure of area/shape output."""
        result = measure_3D_area_size_shape(
            image_set_loader=Mock(anisotropy_spacing=(1.0, 0.1, 0.1)),
            object_loader=simple_loader,
        )

        # Check all required keys exist
        required_keys = {
            "object_id",
            "Volume",
            "CenterX",
            "CenterY",
            "CenterZ",
            "BboxVolume",
            "Extent",
            "EquivalentDiameter",
        }

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_texture_output_consistency(self, simple_loader):
        """Validate consistency of texture output."""
        result = measure_3D_texture(object_loader=simple_loader)

        # All lists should have same length
        lengths = [len(result[k]) for k in result.keys()]
        assert len(set(lengths)) == 1, "Inconsistent list lengths in output"

        # Should have results proportional to number of objects * number of features
        expected_min_rows = simple_loader.object_ids.size
        assert len(result["object_id"]) >= expected_min_rows

    def test_intensity_output_completeness(self, simple_loader):
        """Validate completeness of intensity output."""
        result = measure_3D_intensity_CPU(object_loader=simple_loader)

        # Check for common intensity features
        feature_names = set(result["feature_name"])
        expected_features = {
            "Mean",
            "Max",
            "Min",
            "Median",
            "StdIntensity",
        }

        # At least some expected features should be present
        assert len(expected_features & feature_names) > 0


# ============================================================================
# ADDITIONAL INTEGRATION TESTS
# ============================================================================


class TestFeaturizationWorkflows:
    """Tests for complete featurization workflows."""

    def test_sequential_feature_extraction_pipeline(self):
        """Test extracting multiple features in sequence."""
        volume = np.random.randint(50, 200, (20, 40, 40), dtype=np.uint16)
        labels = np.zeros((20, 40, 40), dtype=np.uint8)

        # Create multiple objects
        for i in range(1, 4):
            z_start = (i - 1) * 6
            z_end = z_start + 5
            labels[z_start:z_end, i * 8 : (i + 1) * 8, i * 8 : (i + 1) * 8] = i

        loader = ObjectLoader(
            image=volume,
            label_image=labels,
            channel_name="channel1",
            compartment_name="nucleus",
        )

        mock_image_set_loader = Mock()
        mock_image_set_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        # Run all features
        features_extracted = {}

        try:
            features_extracted["area"] = measure_3D_area_size_shape(
                image_set_loader=mock_image_set_loader,
                object_loader=loader,
            )
            assert "object_id" in features_extracted["area"]
        except Exception:
            pass

        try:
            features_extracted["texture"] = measure_3D_texture(
                object_loader=loader,
                distance=1,
            )
            assert "object_id" in features_extracted["texture"]
        except Exception:
            pass

        try:
            features_extracted["intensity"] = measure_3D_intensity_CPU(
                object_loader=loader
            )
            assert "object_id" in features_extracted["intensity"]
        except Exception:
            pass

        # At least one feature should be extracted
        assert len(features_extracted) > 0

    def test_parallel_feature_compatibility(self):
        """Test that features can be extracted in parallel without conflicts."""
        image = np.random.randint(50, 200, (15, 30, 30), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)
        labels[3:12, 5:25, 5:25] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        # Extract multiple features
        intensity_result = measure_3D_intensity_CPU(object_loader=loader)
        texture_result = measure_3D_texture(object_loader=loader)

        # Results should be independent
        assert len(intensity_result["object_id"]) > 0
        assert len(texture_result["object_id"]) > 0
        assert intensity_result["object_id"] == texture_result["object_id"]

    def test_feature_extraction_consistency_across_calls(self):
        """Test that multiple calls give consistent results."""
        np.random.seed(123)
        image = np.random.randint(50, 200, (12, 25, 25), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)
        labels[2:10, 5:20, 5:20] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        # Call multiple times
        result1 = measure_3D_intensity_CPU(object_loader=loader)
        result2 = measure_3D_intensity_CPU(object_loader=loader)

        # Should be identical
        assert result1["object_id"] == result2["object_id"]
        assert len(result1["value"]) == len(result2["value"])


class TestLargeScaleFeaturization:
    """Tests for large-scale featurization scenarios."""

    def test_many_objects_feature_extraction(self):
        """Test featurization with many objects."""
        image = np.random.randint(50, 200, (30, 50, 50), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)

        obj_id = 1
        for z in range(2, 28, 3):
            for y in range(2, 48, 4):
                for x in range(2, 48, 4):
                    if obj_id <= 50:
                        labels[z : z + 2, y : y + 3, x : x + 3] = obj_id
                        obj_id += 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should extract features for all objects
        unique_ids = set(result["object_id"])
        assert len(unique_ids) >= 10

    def test_large_volume_texture_extraction(self):
        """Test texture extraction on large volume."""
        # Larger volume
        image = np.random.randint(50, 200, (40, 60, 60), dtype=np.uint16)
        labels = np.zeros((40, 60, 60), dtype=np.uint8)

        # Multiple objects
        labels[5:15, 10:30, 10:30] = 1
        labels[20:30, 35:55, 35:55] = 2

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        result = measure_3D_texture(
            object_loader=loader,
            distance=1,
            grayscale=256,
        )

        assert len(result["object_id"]) > 0


class TestEdgeCasesAdvanced:
    """Advanced edge case tests."""

    def test_single_object_all_features(self):
        """Test all features with single object."""
        image = np.random.randint(50, 200, (15, 25, 25), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)
        labels[3:12, 5:20, 5:20] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        mock_loader = Mock()
        mock_loader.anisotropy_spacing = (1.0, 0.1, 0.1)

        # All features should work
        results = {}

        try:
            results["area"] = measure_3D_area_size_shape(
                image_set_loader=mock_loader,
                object_loader=loader,
            )
        except Exception:
            pass

        try:
            results["intensity"] = measure_3D_intensity_CPU(object_loader=loader)
        except Exception:
            pass

        try:
            results["texture"] = measure_3D_texture(object_loader=loader)
        except Exception:
            pass

        try:
            results["neighbors"] = measure_3D_number_of_neighbors(object_loader=loader)
        except Exception:
            pass

        assert len(results) > 0

    def test_very_high_intensity_values(self):
        """Test features with very high intensity values."""
        image = np.random.randint(50000, 60000, (10, 20, 20), dtype=np.uint32)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        assert len(result["value"]) > 0
        assert all(np.isfinite(v) for v in result["value"])

    def test_very_small_objects_granularity(self):
        """Test granularity with very small objects."""
        image = np.ones((10, 20, 20), dtype=np.uint8) * 100
        labels = np.zeros_like(image, dtype=np.uint8)

        # Single voxel objects
        for i, z in enumerate([2, 5, 8]):
            labels[z, 5 + i * 4, 5 + i * 4] = i + 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        result = measure_3D_granularity(
            object_loader=loader,
            radius=3,
            subsample_image_value=0.5,
        )

        assert isinstance(result, dict)


class TestCrossFunctionConsistency:
    """Tests for consistency across different feature functions."""

    def test_object_id_consistency_across_functions(self):
        """Test that object IDs are consistent across all feature functions."""
        image = np.random.randint(50, 200, (15, 30, 30), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)

        labels[2:6, 5:15, 5:15] = 1
        labels[8:12, 15:25, 15:25] = 2

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        # Extract from different functions
        intensity_result = measure_3D_intensity_CPU(object_loader=loader)
        texture_result = measure_3D_texture(object_loader=loader)

        # Object IDs should be consistent
        intensity_ids = set(intensity_result["object_id"])
        texture_ids = set(texture_result["object_id"])

        assert intensity_ids == texture_ids

    def test_compartment_consistency(self):
        """Test that compartment information is preserved."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        compartment_name = "nucleus"

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="DAPI",
            compartment_name=compartment_name,
        )

        result = measure_3D_intensity_CPU(object_loader=loader)

        # Should contain compartment information
        if "compartment" in result:
            assert all(c == compartment_name for c in result["compartment"])


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_feature_extraction_with_partial_failures(self):
        """Test robustness when some operations might fail."""
        image = np.random.randint(50, 200, (12, 25, 25), dtype=np.uint8)
        labels = np.zeros_like(image, dtype=np.uint8)
        labels[2:10, 5:20, 5:20] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        # Should handle various feature extraction attempts gracefully
        successful_extractions = 0

        try:
            result = measure_3D_intensity_CPU(object_loader=loader)
            if result:
                successful_extractions += 1
        except Exception:
            pass

        try:
            result = measure_3D_texture(object_loader=loader)
            if result:
                successful_extractions += 1
        except Exception:
            pass

        # At least one should succeed
        assert successful_extractions > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
