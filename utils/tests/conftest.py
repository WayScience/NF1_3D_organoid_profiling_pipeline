"""Pytest configuration and shared fixtures for featurization tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# BASIC 3D DATA FIXTURES
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


# ============================================================================
# COLOCALIZATION TEST FIXTURES
# ============================================================================


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


@pytest.fixture
def colocalized_vs_separate_signals():
    """Create colocalized and non-colocalized signal pairs."""
    # Channel 1
    img1 = np.zeros((15, 30, 30), dtype=np.uint8)
    # Channel 2
    img2 = np.zeros((15, 30, 30), dtype=np.uint8)

    # Perfect colocalization region (overlap = 100%)
    img1[3:12, 3:12, 3:12] = 200
    img2[3:12, 3:12, 3:12] = 180

    # Partial colocalization (50% overlap)
    img1[3:12, 15:25, 3:12] = 150
    img2[6:15, 15:25, 3:12] = 140

    # No colocalization (separate regions)
    img1[3:10, 3:10, 18:25] = 160
    img2[10:15, 20:27, 18:25] = 170

    return img1, img2


# ============================================================================
# FEATURE-SPECIFIC TEST DATA FIXTURES
# ============================================================================


@pytest.fixture
def size_varied_objects():
    """Create small, medium, and large objects for area/size/shape testing."""
    image = np.random.randint(50, 200, (30, 40, 40), dtype=np.uint8)
    labels = np.zeros((30, 40, 40), dtype=np.uint8)

    # Small object: 3x3x3 = 27 voxels
    labels[5:8, 5:8, 5:8] = 1

    # Medium object: 8x8x8 = 512 voxels
    labels[10:18, 10:18, 10:18] = 2

    # Large object: 15x15x15 = 3375 voxels
    labels[5:20, 20:35, 20:35] = 3

    return image, labels


@pytest.fixture
def granularity_test_data():
    """Create objects with high, medium, and low granularity."""
    labels = np.zeros((20, 40, 40), dtype=np.uint8)
    image = np.zeros((20, 40, 40), dtype=np.uint16)

    # High granularity: many small bright spots (textured, grainy)
    labels[2:18, 2:18, 2:18] = 1
    for _ in range(50):
        z, y, x = np.random.randint(2, 18, 3)
        # Create small 2x2x2 bright spots
        image[z : z + 2, y : y + 2, x : x + 2] = np.random.randint(800, 1000)
    # Fill background with low intensity
    image[labels == 1] = np.where(image[labels == 1] > 0, image[labels == 1], 100)

    # Medium granularity: moderate-sized structures
    labels[2:18, 20:38, 2:18] = 2
    for _ in range(15):
        z, y, x = (
            np.random.randint(2, 18),
            np.random.randint(20, 35),
            np.random.randint(2, 15),
        )
        # Create medium 4x4x4 bright regions
        image[z : z + 4, y : y + 4, x : x + 4] = np.random.randint(600, 800)
    image[labels == 2] = np.where(image[labels == 2] > 0, image[labels == 2], 120)

    # Low granularity: smooth, uniform regions (not grainy)
    labels[2:18, 20:38, 20:38] = 3
    # Smooth gradient instead of spots
    z_grid, y_grid, x_grid = np.mgrid[2:18, 20:38, 20:38]
    image[2:18, 20:38, 20:38] = (
        150 + 30 * np.sin((z_grid - 2) * 0.3) + 20 * np.cos((y_grid - 20) * 0.2)
    )

    return image, labels


@pytest.fixture
def texture_varied_objects():
    """Create objects with uniform and non-uniform texture."""
    image = np.zeros((20, 40, 40), dtype=np.uint16)
    labels = np.zeros((20, 40, 40), dtype=np.uint8)

    # Uniform texture: constant intensity
    labels[3:17, 3:17, 3:17] = 1
    image[labels == 1] = 500  # Perfectly uniform

    # Non-uniform texture: random noise
    labels[3:17, 20:37, 3:17] = 2
    np.random.seed(42)
    image[labels == 2] = np.random.randint(300, 700, size=np.sum(labels == 2))

    # Structured texture: checkerboard pattern
    labels[3:17, 20:37, 20:37] = 3
    z_idx, y_idx, x_idx = np.where(labels == 3)
    # Create alternating pattern
    for z, y, x in zip(z_idx, y_idx, x_idx):
        image[z, y, x] = 600 if (z + y + x) % 2 == 0 else 400

    return image, labels


@pytest.fixture
def intensity_varied_objects():
    """Create high, medium, and low intensity objects."""
    labels = np.zeros((20, 40, 40), dtype=np.uint8)
    image = np.zeros((20, 40, 40), dtype=np.uint16)

    # High intensity object
    labels[3:17, 3:17, 3:17] = 1
    image[labels == 1] = np.random.randint(3000, 4000, size=np.sum(labels == 1))

    # Medium intensity object
    labels[3:17, 20:37, 3:17] = 2
    image[labels == 2] = np.random.randint(1000, 2000, size=np.sum(labels == 2))

    # Low intensity object
    labels[3:17, 20:37, 20:37] = 3
    image[labels == 3] = np.random.randint(100, 500, size=np.sum(labels == 3))

    return image, labels


@pytest.fixture
def neighbor_distributions():
    """Create different spatial distributions for neighbor testing."""
    image = np.random.randint(50, 200, (30, 60, 60), dtype=np.uint8)
    labels = np.zeros((30, 60, 60), dtype=np.uint8)

    # Many close neighbors (organoid-like cluster)
    obj_id = 1
    # Dense cluster in corner
    for z_offset in range(0, 12, 4):
        for y_offset in range(0, 12, 4):
            for x_offset in range(0, 12, 4):
                z_start = 5 + z_offset
                y_start = 5 + y_offset
                x_start = 5 + x_offset
                labels[
                    z_start : z_start + 3, y_start : y_start + 3, x_start : x_start + 3
                ] = obj_id
                obj_id += 1

    # Few distant neighbors (scattered)
    # Well-separated objects
    labels[10:13, 40:43, 40:43] = obj_id
    obj_id += 1
    labels[20:23, 50:53, 50:53] = obj_id
    obj_id += 1
    labels[25:28, 30:33, 45:48] = obj_id
    obj_id += 1

    # Organoid-like ring distribution (hollow sphere)
    center = (15, 40, 25)
    radius_outer = 8
    radius_inner = 5
    for z in range(center[0] - radius_outer, center[0] + radius_outer):
        for y in range(center[1] - radius_outer, center[1] + radius_outer):
            for x in range(center[2] - radius_outer, center[2] + radius_outer):
                if 0 <= z < 30 and 0 <= y < 60 and 0 <= x < 60:
                    dist = np.sqrt(
                        (z - center[0]) ** 2
                        + (y - center[1]) ** 2
                        + (x - center[2]) ** 2
                    )
                    if radius_inner < dist <= radius_outer:
                        labels[z, y, x] = obj_id
        obj_id += 1

    return image, labels


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================


@pytest.fixture
def realistic_3d_volume():
    """Create a more realistic 3D volume for integration tests."""
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
def realistic_label_image():
    """Create a realistic label image for integration tests."""
    labels = np.zeros((30, 50, 50), dtype=np.uint8)

    # Object 1
    labels[5:15, 10:20, 10:20] = 1

    # Object 2
    labels[10:20, 25:40, 25:40] = 2

    # Object 3
    labels[15:25, 5:15, 35:50] = 3

    return labels


# ============================================================================
# PERFORMANCE TEST FIXTURES
# ============================================================================


@pytest.fixture
def large_volume():
    """Create a larger volume for performance testing."""
    return np.random.randint(50, 200, (50, 100, 100), dtype=np.uint8)


@pytest.fixture
def large_label_image():
    """Create larger label image for performance testing."""
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
