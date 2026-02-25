"""Tests for image_utils functions."""

from __future__ import annotations

import numpy as np
import pytest
from image_analysis_3D.image_utils.image_utils import (
    check_for_xy_squareness,
    crop_3D_image,
    expand_box,
    new_crop_border,
    select_objects_from_label,
    single_3D_image_expand_bbox,
    square_off_xy_crop_bbox,
)


class TestSelectObjectsFromLabel:
    """Test select_objects_from_label function."""

    def test_select_single_object(self) -> None:
        """Test selecting a single object from label image."""
        label_image = np.array([[[1, 1, 0], [0, 2, 2], [3, 3, 3]]])
        result = select_objects_from_label(label_image, [2])
        expected = np.array([[[0, 0, 0], [0, 2, 2], [0, 0, 0]]])
        np.testing.assert_array_equal(result, expected)

    def test_select_multiple_objects(self) -> None:
        """Test selecting multiple objects from label image."""
        label_image = np.array([[[1, 1, 0], [0, 2, 2], [3, 3, 3]]])
        result = select_objects_from_label(label_image, [1, 3])
        expected = np.array([[[1, 1, 0], [0, 0, 0], [3, 3, 3]]])
        np.testing.assert_array_equal(result, expected)

    def test_select_nonexistent_object(self) -> None:
        """Test selecting object that doesn't exist."""
        label_image = np.array([[[1, 1, 0], [0, 2, 2], [3, 3, 3]]])
        result = select_objects_from_label(label_image, [99])
        expected = np.zeros_like(label_image)
        np.testing.assert_array_equal(result, expected)

    def test_original_image_unchanged(self) -> None:
        """Test that the original image is not modified."""
        original = np.array([[[1, 1, 0], [0, 2, 2], [3, 3, 3]]])
        label_image = original.copy()
        _ = select_objects_from_label(label_image, [2])
        np.testing.assert_array_equal(label_image, original)


class TestExpandBox:
    """Test expand_box function."""

    def test_expansion_prioritizes_min_side(self) -> None:
        """Test that expansion prioritizes minimum side first."""
        result = expand_box(
            min_coor=0, max_coord=10, current_min=4, current_max=6, expand_by=4
        )
        # Should expand min side first: 4→3→2→1→0
        assert result == (0, 6)

    def test_expansion_at_lower_boundary(self) -> None:
        """Test expansion when already at lower boundary."""
        result = expand_box(
            min_coor=0, max_coord=10, current_min=0, current_max=2, expand_by=4
        )
        # Can't expand min, so expands max: 2→3→4→5→6
        assert result == (0, 6)

    def test_expansion_at_upper_boundary(self) -> None:
        """Test expansion when already at upper boundary."""
        result = expand_box(
            min_coor=0, max_coord=10, current_min=8, current_max=10, expand_by=4
        )
        # Expands min side: 8→7→6→5→4
        assert result == (4, 10)

    def test_no_expansion_needed(self) -> None:
        """Test when no expansion is requested."""
        result = expand_box(
            min_coor=0, max_coord=10, current_min=3, current_max=7, expand_by=0
        )
        assert result == (3, 7)

    def test_insufficient_space_for_expansion(self) -> None:
        """Test error when requested expansion is not possible."""
        result = expand_box(
            min_coor=0, max_coord=10, current_min=2, current_max=8, expand_by=10
        )
        assert isinstance(result, ValueError)


class TestNewCropBorder:
    """Test new_crop_border function."""

    def test_expand_first_bbox(self) -> None:
        """Test expanding first bbox to match second."""
        image = np.zeros((20, 20, 20))
        bbox1 = (5, 5, 5, 8, 8, 8)  # 3x3x3
        bbox2 = (10, 10, 10, 15, 15, 15)  # 5x5x5
        result1, result2 = new_crop_border(bbox1, bbox2, image)

        # bbox1 should be expanded
        z1, y1, x1, z2, y2, x2 = result1
        assert (z2 - z1) == (y2 - y1) == (x2 - x1) == 5

    def test_expand_second_bbox(self) -> None:
        """Test expanding second bbox to match first."""
        image = np.zeros((20, 20, 20))
        bbox1 = (5, 5, 5, 10, 10, 10)  # 5x5x5
        bbox2 = (12, 12, 12, 15, 15, 15)  # 3x3x3
        result1, result2 = new_crop_border(bbox1, bbox2, image)

        # bbox2 should be expanded
        z1, y1, x1, z2, y2, x2 = result2
        assert (z2 - z1) == (y2 - y1) == (x2 - x1) == 5

    def test_equal_sized_bboxes(self) -> None:
        """Test when bboxes are already equal."""
        image = np.zeros((20, 20, 20))
        bbox1 = (5, 5, 5, 10, 10, 10)
        bbox2 = (12, 12, 12, 17, 17, 17)
        result1, result2 = new_crop_border(bbox1, bbox2, image)

        assert result1 == bbox1
        assert result2 == bbox2


class TestCrop3DImage:
    """Test crop_3D_image function."""

    def test_basic_crop(self) -> None:
        """Test basic cropping of 3D image."""
        image = np.arange(1000).reshape(10, 10, 10)
        bbox = (2, 3, 4, 5, 6, 7)
        result = crop_3D_image(image, bbox)

        assert result.shape == (3, 3, 3)
        np.testing.assert_array_equal(result, image[2:5, 3:6, 4:7])

    def test_crop_full_image(self) -> None:
        """Test cropping with bbox encompassing entire image."""
        image = np.ones((5, 5, 5))
        bbox = (0, 0, 0, 5, 5, 5)
        result = crop_3D_image(image, bbox)

        np.testing.assert_array_equal(result, image)

    def test_crop_single_voxel(self) -> None:
        """Test cropping to a single voxel."""
        image = np.arange(27).reshape(3, 3, 3)
        bbox = (1, 1, 1, 2, 2, 2)
        result = crop_3D_image(image, bbox)

        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == image[1, 1, 1]


class TestSingle3DImageExpandBbox:
    """Test single_3D_image_expand_bbox function."""

    def test_basic_expansion_isotropic(self) -> None:
        """Test basic expansion with isotropic voxels (anisotropy=1)."""
        image = np.zeros((100, 100, 100))
        bbox = (40, 40, 40, 60, 60, 60)
        result = single_3D_image_expand_bbox(
            image, bbox, expand_pixels=10, anisotropy_factor=1
        )

        assert result == (30, 30, 30, 70, 70, 70)

    def test_expansion_with_anisotropy(self) -> None:
        """Test expansion with anisotropic voxels."""
        image = np.zeros((50, 100, 100))
        bbox = (20, 40, 40, 30, 60, 60)
        result = single_3D_image_expand_bbox(
            image, bbox, expand_pixels=10, anisotropy_factor=5
        )

        # Z dimension should be compressed due to anisotropy
        z1, y1, x1, z2, y2, x2 = result
        assert y1 == 30 and y2 == 70
        assert x1 == 30 and x2 == 70
        # Z should be less expanded due to anisotropy factor
        assert z2 - z1 < y2 - y1

    def test_expansion_at_image_boundary(self) -> None:
        """Test that expansion respects image boundaries."""
        image = np.zeros((20, 20, 20))
        bbox = (2, 2, 2, 5, 5, 5)
        result = single_3D_image_expand_bbox(
            image, bbox, expand_pixels=50, anisotropy_factor=1
        )

        z1, y1, x1, z2, y2, x2 = result
        assert z1 >= 0 and z2 <= 20
        assert y1 >= 0 and y2 <= 20
        assert x1 >= 0 and x2 <= 20

    def test_no_expansion(self) -> None:
        """Test with zero pixel expansion."""
        image = np.zeros((50, 50, 50))
        bbox = (10, 15, 20, 30, 35, 40)
        result = single_3D_image_expand_bbox(
            image, bbox, expand_pixels=0, anisotropy_factor=1
        )

        assert result == bbox


class TestCheckForXySquareness:
    """Test check_for_xy_squareness function."""

    def test_perfect_square(self) -> None:
        """Test bbox that is perfectly square in xy plane."""
        bbox = (0, 10, 10, 5, 20, 20)  # z irrelevant, y=10, x=10
        result = check_for_xy_squareness(bbox)
        assert result == 1.0

    def test_wider_than_tall(self) -> None:
        """Test bbox that is wider (x) than tall (y)."""
        bbox = (0, 10, 10, 5, 15, 20)  # y=5, x=10
        result = check_for_xy_squareness(bbox)
        assert result == 0.5

    def test_taller_than_wide(self) -> None:
        """Test bbox that is taller (y) than wide (x)."""
        bbox = (0, 10, 10, 5, 20, 15)  # y=10, x=5
        result = check_for_xy_squareness(bbox)
        assert result == 2.0

    def test_zero_width_raises_error(self) -> None:
        """Test that zero width in x dimension raises ValueError."""
        bbox = (0, 10, 10, 5, 20, 10)  # x_length = 0
        with pytest.raises(ValueError, match="Cannot compute xy squareness"):
            check_for_xy_squareness(bbox)


class TestSquareOffXyCropBbox:
    """Test square_off_xy_crop_bbox function."""

    def test_already_square(self) -> None:
        """Test bbox that is already square."""
        bbox = (5, 10, 10, 15, 20, 20)
        result = square_off_xy_crop_bbox(bbox)
        assert result == bbox

    def test_expand_y_dimension(self) -> None:
        """Test expanding y dimension to match x."""
        bbox = (5, 10, 10, 15, 15, 20)  # y=5, x=10
        result = square_off_xy_crop_bbox(bbox)

        zmin, ymin, xmin, zmax, ymax, xmax = result
        assert zmin == 5 and zmax == 15
        assert xmin == 10 and xmax == 20
        assert ymax - ymin == xmax - xmin  # Should be square

    def test_expand_x_dimension(self) -> None:
        """Test expanding x dimension to match y."""
        bbox = (5, 10, 10, 15, 20, 15)  # y=10, x=5
        result = square_off_xy_crop_bbox(bbox)

        zmin, ymin, xmin, zmax, ymax, xmax = result
        assert zmin == 5 and zmax == 15
        assert ymin == 10 and ymax == 20
        assert ymax - ymin == xmax - xmin  # Should be square

    def test_centered_expansion(self) -> None:
        """Test that expansion is centered."""
        bbox = (0, 10, 10, 10, 14, 20)  # y=4, x=10, needs +3 on each side of y
        result = square_off_xy_crop_bbox(bbox)

        _, ymin, xmin, _, ymax, xmax = result
        assert ymin == 7  # 10 - 3
        assert ymax == 17  # 14 + 3
        assert ymax - ymin == xmax - xmin

    def test_z_dimension_unchanged(self) -> None:
        """Test that z dimension is never modified."""
        bbox = (5, 10, 10, 15, 15, 25)
        result = square_off_xy_crop_bbox(bbox)

        assert result[0] == 5
        assert result[3] == 15


class TestImageUtilsEdgeCases:
    """Test edge cases and error conditions."""

    def test_expand_box_with_floats(self) -> None:
        """Test expand_box handles float coordinates."""
        result = expand_box(
            min_coor=0.0, max_coord=10.0, current_min=4.0, current_max=6.0, expand_by=2
        )
        assert result == (2.0, 8.0)

    def test_crop_with_float_bbox(self) -> None:
        """Test crop_3D_image with float bbox values."""
        image = np.arange(1000).reshape(10, 10, 10)
        bbox = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        result = crop_3D_image(image, bbox)

        # Floats should work as indices get casted
        assert result.shape == (3, 3, 3)

    def test_empty_bbox_dimensions(self) -> None:
        """Test behavior with zero-sized dimensions."""
        image = np.zeros((10, 10, 10))
        bbox = (5, 5, 5, 5, 8, 8)  # z dimension is zero
        result = crop_3D_image(image, bbox)

        assert result.shape == (0, 3, 3)

    def test_large_anisotropy_factor(self) -> None:
        """Test with large anisotropy factor."""
        image = np.zeros((10, 100, 100))  # Very anisotropic
        bbox = (2, 40, 40, 8, 60, 60)
        result = single_3D_image_expand_bbox(
            image, bbox, expand_pixels=20, anisotropy_factor=10
        )

        # Should handle large anisotropy without error
        z1, y1, x1, z2, y2, x2 = result
        assert z1 >= 0 and z2 <= 10
        assert y1 >= 0 and y2 <= 100
        assert x1 >= 0 and x2 <= 100
