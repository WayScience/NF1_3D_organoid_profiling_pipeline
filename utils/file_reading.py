import numpy as np
import skimage
import tifffile


def read_zstack_image(file_path: str) -> np.ndarray:
    """
    Description
    -----------
    Reads in a z-stack image from a given file path and returns it as a numpy array.

    Parameters
    ----------
    file_path : str
        The path to the z-stack image file.
    Returns
    -------
    np.ndarray
        The z-stack image as a numpy array.

    Raises
    -------
    ValueError
        If the image has less than 3 dimensions.
    """

    img = tifffile.imread(file_path)

    if len(img.shape) > 5:
        # determine in any of the dimensions is size of 1?
        img = np.squeeze(img)
    elif len(img.shape) < 3:
        raise ValueError(f"Image at {file_path} has less than 3 dimensions")

    if img.dtype != np.uint16:
        if img.dtype in [np.float32, np.float64]:
            # For float images, first rescale to 0-1 range, then convert
            img = skimage.exposure.rescale_intensity(img, out_range=(0, 1))
            img = skimage.img_as_uint(img)
        else:
            # For other integer types
            img = skimage.img_as_uint(img)

    return img
