"""Helpers for building channel montages and composites."""

import numpy as np


def add_scale_bar(
    image: np.ndarray,
    pixel_size_microns: float,
    bar_length_microns: float = 10,
    bar_thickness: int = 10,
    padding_pixels: int = 10,
) -> np.ndarray:
    """
    Add a scale bar to the bottom right corner of the image.

    Parameters
    ----------
    image : np.ndarray
        2D (grayscale) or 3D (RGB) array representing the image.
    pixel_size_microns : float
        Size of one pixel in microns.
    bar_length_microns : float, optional
        Desired length of the scale bar in microns, by default 10 microns.
    bar_thickness : int, optional
        Thickness of the scale bar in pixels, by default 10 pixels.
    padding_pixels : int, optional
        Padding from the image edges in pixels, by default 10 pixels.

    Returns
    -------
    np.ndarray
        Copy of the input image with the scale bar added. For grayscale
        images the bar is drawn as an over-range value (max + 1); for RGB
        images it is drawn as white (255).

    Raises
    ------
    ValueError
        If the image is too small to fit a scale bar with the specified
        padding and thickness.
    """
    # Calculate the length of the scale bar in pixels
    bar_length_pixels = int(round(bar_length_microns / pixel_size_microns))

    # Create a copy of the image to draw on
    image_with_bar = np.array(image, copy=True)

    # Define the position for the scale bar (bottom right corner with some padding)
    start_x = image.shape[1] - padding_pixels - bar_length_pixels
    start_y = image.shape[0] - padding_pixels - bar_thickness

    if start_x < 0 or start_y < 0:
        raise ValueError(
            "Image is too small to add a scale bar with the specified padding and thickness."
        )

    # Draw the scale bar as white for RGB images, or as an over-range value for grayscale images
    if image_with_bar.ndim == 2:
        image_with_bar = image_with_bar.astype(np.float32, copy=False)
        image_with_bar[
            start_y : start_y + bar_thickness, start_x : start_x + bar_length_pixels
        ] = image_with_bar.max() + 1
    else:
        image_with_bar[
            start_y : start_y + bar_thickness, start_x : start_x + bar_length_pixels
        ] = 255

    return image_with_bar
