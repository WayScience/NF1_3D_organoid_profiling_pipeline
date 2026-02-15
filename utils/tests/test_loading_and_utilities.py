"""Tests for loading classes, errors, and utility functions."""

from __future__ import annotations

import numpy as np
import pytest
import skimage.io
from src.image_analysis_3D.featurization_utils.errors import ProcessorTypeError
from src.image_analysis_3D.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
    remove_underscores_from_string,
)
from src.image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)

# ============================================================================
# TESTS: loading_classes.py
# ============================================================================


class TestImageSetLoader:
    """Tests for ImageSetLoader class."""

    @pytest.fixture
    def temp_image_dir(self, tmp_path):
        """Create temporary directory with test images."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        # Create test images
        test_image = np.random.randint(0, 255, (10, 20, 20), dtype=np.uint8)
        test_mask = np.zeros((10, 20, 20), dtype=np.uint8)
        test_mask[2:8, 5:15, 5:15] = 1
        test_mask[3:7, 10:15, 10:15] = 2

        skimage.io.imsave(str(image_dir / "DAPI.tif"), test_image)
        skimage.io.imsave(str(image_dir / "GFP.tif"), test_image)
        skimage.io.imsave(str(mask_dir / "nuclei.tif"), test_mask)
        skimage.io.imsave(str(mask_dir / "cells.tif"), test_mask)

        return {
            "image_dir": image_dir,
            "mask_dir": mask_dir,
            "image": test_image,
            "mask": test_mask,
        }

    def test_initialization_basic(self, temp_image_dir):
        """Test basic initialization of ImageSetLoader."""
        loader = ImageSetLoader(
            image_set_path=temp_image_dir["image_dir"],
            mask_set_path=temp_image_dir["mask_dir"],
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={
                "DAPI": "DAPI",
                "GFP": "GFP",
                "nuclei": "nuclei",
                "cells": "cells",
            },
            image_set_name="test_well",
        )

        assert loader.image_set_name == "test_well"
        assert loader.anisotropy_spacing == (1.0, 0.1, 0.1)
        assert loader.anisotropy_factor == 10.0

    def test_initialization_with_none_path(self):
        """Test initialization with None image path."""
        loader = ImageSetLoader(
            image_set_path=None,
            mask_set_path=None,
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={},
            image_set_name="test",
        )

        assert loader.image_set_name == "test"
        assert loader.anisotropy_factor == 10.0

    def test_anisotropy_calculation(self, temp_image_dir):
        """Test anisotropy factor calculation."""
        # Different spacing ratios
        test_cases = [
            ((2.0, 0.1, 0.1), 20.0),
            ((1.0, 1.0, 1.0), 1.0),
            ((0.5, 0.1, 0.1), 5.0),
        ]

        for spacing, expected_factor in test_cases:
            loader = ImageSetLoader(
                image_set_path=temp_image_dir["image_dir"],
                mask_set_path=temp_image_dir["mask_dir"],
                anisotropy_spacing=spacing,
                channel_mapping={"DAPI": "DAPI"},
                image_set_name="test",
            )
            assert loader.anisotropy_factor == expected_factor

    def test_get_image_names(self, temp_image_dir):
        """Test retrieval of image names."""
        loader = ImageSetLoader(
            image_set_path=temp_image_dir["image_dir"],
            mask_set_path=temp_image_dir["mask_dir"],
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={"DAPI": "DAPI", "GFP": "GFP"},
            image_set_name="test_well",
        )

        names = loader.get_image_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_compartments(self, temp_image_dir):
        """Test retrieval of compartment names."""
        loader = ImageSetLoader(
            image_set_path=temp_image_dir["image_dir"],
            mask_set_path=temp_image_dir["mask_dir"],
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={"nuclei": "nuclei", "cells": "cells"},
            image_set_name="test_well",
        )

        compartments = loader.get_compartments()
        assert isinstance(compartments, list)

    def test_get_anisotropy(self, temp_image_dir):
        """Test anisotropy retrieval."""
        loader = ImageSetLoader(
            image_set_path=temp_image_dir["image_dir"],
            mask_set_path=temp_image_dir["mask_dir"],
            anisotropy_spacing=(2.0, 0.2, 0.2),
            channel_mapping={"DAPI": "DAPI"},
            image_set_name="test",
        )

        factor = loader.get_anisotropy()
        assert factor == 10.0

    def test_retrieve_image_attributes(self, temp_image_dir):
        """Test attribute retrieval from images."""
        loader = ImageSetLoader(
            image_set_path=temp_image_dir["image_dir"],
            mask_set_path=temp_image_dir["mask_dir"],
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={"nuclei": "nuclei"},
            image_set_name="test",
        )

        loader.retrieve_image_attributes()
        assert hasattr(loader, "unique_mask_objects")

    def test_empty_channel_mapping(self, temp_image_dir):
        """Test with empty channel mapping."""
        loader = ImageSetLoader(
            image_set_path=temp_image_dir["image_dir"],
            mask_set_path=temp_image_dir["mask_dir"],
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={},
            image_set_name="test",
        )

        assert loader.image_set_name == "test"


class TestObjectLoader:
    """Tests for ObjectLoader class."""

    @pytest.fixture
    def test_data(self):
        """Create test data for ObjectLoader."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:5, 5:10, 5:10] = 1
        labels[5:8, 10:15, 10:15] = 2
        return image, labels

    def test_initialization(self, test_data):
        """Test basic ObjectLoader initialization."""
        image, labels = test_data
        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="DAPI",
            compartment_name="Nuclei",
        )

        assert loader.channel == "DAPI"
        assert loader.compartment == "Nuclei"
        assert len(loader.object_ids) > 0

    def test_object_id_extraction(self, test_data):
        """Test that object IDs are correctly extracted."""
        image, labels = test_data
        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        # Should have 2 objects
        assert len(loader.object_ids) == 2
        assert 1 in loader.object_ids
        assert 2 in loader.object_ids

    def test_empty_label_image(self):
        """Test with empty (all zeros) label image."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        assert len(loader.object_ids) == 0

    def test_single_object(self):
        """Test with single object."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:8, 5:15, 5:15] = 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        assert len(loader.object_ids) == 1
        assert loader.object_ids[0] == 1

    def test_many_objects(self):
        """Test with many objects."""
        image = np.random.randint(50, 200, (10, 20, 20), dtype=np.uint8)
        labels = np.zeros((10, 20, 20), dtype=np.uint8)

        # Create 10 objects
        obj_id = 1
        for z in range(0, 10, 2):
            labels[z : z + 1, 5:10, 5:10] = obj_id
            obj_id += 1

        loader = ObjectLoader(
            image=image,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )

        assert len(loader.object_ids) == 5

    def test_different_data_types(self):
        """Test with different image data types."""
        labels = np.zeros((10, 20, 20), dtype=np.uint8)
        labels[2:5, 5:10, 5:10] = 1

        # Test uint8
        image_uint8 = np.random.randint(0, 255, (10, 20, 20), dtype=np.uint8)
        loader = ObjectLoader(
            image=image_uint8,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )
        assert len(loader.object_ids) > 0

        # Test uint16
        image_uint16 = np.random.randint(0, 65535, (10, 20, 20), dtype=np.uint16)
        loader = ObjectLoader(
            image=image_uint16,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )
        assert len(loader.object_ids) > 0

        # Test float32
        image_float = np.random.rand(10, 20, 20).astype(np.float32)
        loader = ObjectLoader(
            image=image_float,
            label_image=labels,
            channel_name="test",
            compartment_name="test",
        )
        assert len(loader.object_ids) > 0


# ============================================================================
# TESTS: errors.py
# ============================================================================


class TestProcessorTypeError:
    """Tests for ProcessorTypeError exception."""

    def test_error_message(self):
        """Test that error returns correct message."""
        error = ProcessorTypeError()
        assert "Processor type not recognized" in str(error)
        assert "CPU" in str(error)
        assert "GPU" in str(error)

    def test_raising_error(self):
        """Test raising the exception."""
        with pytest.raises(ProcessorTypeError) as exc_info:
            raise ProcessorTypeError()

        assert "Processor type not recognized" in str(exc_info.value)

    def test_catching_error(self):
        """Test catching the exception."""
        try:
            raise ProcessorTypeError()
        except ProcessorTypeError as e:
            assert isinstance(e, ProcessorTypeError)
            assert "CPU" in str(e)
        else:
            pytest.fail("Exception not raised")


# ============================================================================
# TESTS: feature_writing_utils.py
# ============================================================================


class TestFeatureWritingUtils:
    """Tests for feature writing utility functions."""

    def test_remove_underscores_basic(self):
        """Test basic underscore removal."""
        assert remove_underscores_from_string("test_string") == "test-string"
        assert remove_underscores_from_string("another_test") == "another-test"

    def test_remove_underscores_multiple_delimiters(self):
        """Test removal of multiple delimiter types."""
        test_cases = [
            ("test_string.value", "test-string-value"),
            ("path/to/file", "path-to-file"),
            ("spaced value", "spaced-value"),
            ("mixed_test.file/path value", "mixed-test-file-path-value"),
        ]

        for input_str, expected in test_cases:
            assert remove_underscores_from_string(input_str) == expected

    def test_remove_underscores_no_delimiters(self):
        """Test with strings that have no delimiters."""
        assert remove_underscores_from_string("nodeli meters") == "nodeli-meters"
        assert remove_underscores_from_string("simple") == "simple"

    def test_remove_underscores_only_delimiters(self):
        """Test with strings of only delimiters."""
        assert remove_underscores_from_string("___") == "---"
        assert remove_underscores_from_string("...") == "---"
        assert remove_underscores_from_string("   ") == "---"

    def test_remove_underscores_empty_string(self):
        """Test with empty string."""
        assert remove_underscores_from_string("") == ""

    def test_format_morphology_feature_name_basic(self):
        """Test basic feature name formatting."""
        result = format_morphology_feature_name(
            compartment="Nuclei",
            channel="DAPI",
            feature_type="Intensity",
            measurement="Mean",
        )

        assert result == "Nuclei_DAPI_Intensity_Mean"

    def test_format_morphology_feature_name_with_underscores(self):
        """Test formatting with underscores in inputs."""
        result = format_morphology_feature_name(
            compartment="Whole_Cell",
            channel="GFP_Channel",
            feature_type="Area_Shape",
            measurement="Total_Area",
        )

        assert result == "Whole-Cell_GFP-Channel_Area-Shape_Total-Area"

    def test_format_morphology_feature_name_with_mixed_delimiters(self):
        """Test formatting with various delimiters."""
        result = format_morphology_feature_name(
            compartment="Outer.Cell",
            channel="Red/Channel",
            feature_type="Texture Feature",
            measurement="Haralick.Contrast",
        )

        assert result == "Outer-Cell_Red-Channel_Texture-Feature_Haralick-Contrast"

    def test_format_morphology_feature_name_consistency(self):
        """Test that same inputs always produce same output."""
        result1 = format_morphology_feature_name("Nuclei", "DAPI", "Intensity", "Mean")
        result2 = format_morphology_feature_name("Nuclei", "DAPI", "Intensity", "Mean")

        assert result1 == result2

    def test_format_morphology_feature_name_all_types(self):
        """Test various feature types."""
        test_cases = [
            ("Cell", "DAPI", "AreaShape", "Volume", "Cell_DAPI_AreaShape_Volume"),
            ("Nuclei", "GFP", "Texture", "Contrast", "Nuclei_GFP_Texture_Contrast"),
            ("Cytoplasm", "RFP", "Intensity", "Max", "Cytoplasm_RFP_Intensity_Max"),
            (
                "Organoid",
                "Brightfield",
                "Granularity",
                "Spectrum",
                "Organoid_Brightfield_Granularity_Spectrum",
            ),
        ]

        for compartment, channel, feature_type, measurement, expected in test_cases:
            result = format_morphology_feature_name(
                compartment, channel, feature_type, measurement
            )
            assert result == expected


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestLoadingIntegration:
    """Integration tests for loading classes."""

    def test_image_set_to_object_loader_workflow(self, tmp_path):
        """Test complete workflow from ImageSetLoader to ObjectLoader."""
        # Create test data
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        test_image = np.random.randint(0, 255, (10, 20, 20), dtype=np.uint8)
        test_mask = np.zeros((10, 20, 20), dtype=np.uint8)
        test_mask[2:5, 5:10, 5:10] = 1

        skimage.io.imsave(str(image_dir / "DAPI.tif"), test_image)
        skimage.io.imsave(str(mask_dir / "nuclei.tif"), test_mask)

        # Create ImageSetLoader
        image_loader = ImageSetLoader(
            image_set_path=image_dir,
            mask_set_path=mask_dir,
            anisotropy_spacing=(1.0, 0.1, 0.1),
            channel_mapping={"DAPI": "DAPI", "nuclei": "nuclei"},
            image_set_name="test_well",
        )

        # Verify it has anisotropy
        assert image_loader.anisotropy_factor == 10.0

        # Create ObjectLoader from ImageSetLoader data
        object_loader = ObjectLoader(
            image=test_image,
            label_image=test_mask,
            channel_name="DAPI",
            compartment_name="Nuclei",
        )

        # Verify both loaders work together
        assert len(object_loader.object_ids) == 1
        assert image_loader.image_set_name == "test_well"
