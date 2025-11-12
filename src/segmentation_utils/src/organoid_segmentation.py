import cellpose.models as models
import numpy as np
import skimage
import torch


def segment_with_diameter(
    img: np.ndarray,
    model: models.Cellpose,
    diameter: int,
    z_axis: int = 0,
    channels: tuple = [1, 0],
    min_diameter: int = 200,  # in pixels, default is 200
    diameter_step: int = 200,
) -> tuple:
    """
    Recursively perform segmentation, stepping down through diameters by 200
    until a valid label is found or the minimum diameter is reached.
    This effectively performs a dynamic search for the largest detectable object
    in the image.

    Parameters
    ----------
    img : np.ndarray
        The image to segment. Can be 3D in the format of (z, y, x).
    model : models.Cellpose
        The Cellpose model to use for segmentation.
    diameter : int
        The diameter to use for segmentation.
        This is where the search starts.
    z_axis : int, optional
        The axis of the z-stack. Default is axis 0.
    channels : tuple, optional
        The channels to use for segmentation. Default is (1, 0).
        Where 1 is the channel for the cytoplasm and 0 using no other channel.
    min_diameter : int, optional
        The minimum diameter to use for segmentation.
        If the diameter is less than this, the function will return empty labels.
        Default is 200 pixels.
    diameter_step : int, optional
        The step size to decrease the diameter by when no labels are found.
        Default is 200 pixels.

    Returns
    -------
    tuple
        labels : np.ndarray
            The labels of the segmented image.
        details : dict
            The details of the segmentation.
        _ : None
            Placeholder for additional return values.
    """

    if diameter < min_diameter:
        print("Minimum diameter reached. Returning empty labels.")
        zero_labels = np.zeros_like(img)
        return zero_labels, None, None

    labels, details, _ = model.eval(
        img, channels=channels, z_axis=z_axis, diameter=diameter
    )

    if labels is None:
        print(f"Labels are empty for diameter {diameter}. Trying smaller diameter...")
        return segment_with_diameter(
            img, model, channels, z_axis, diameter - diameter_step
        )

    return labels, details, _


# test the function with three cylinders each with a different diameter
def test_segment_with_diameter():
    diameters = [100, 250, 400, 600, 800, 1000]
    z_depth = 1  # No z-depth for 2D images
    for diameter in diameters:
        img = np.zeros((1500, 1500), dtype=np.uint8)
        rr, cc = skimage.draw.disk((500, 500), diameter / 2)
        img[rr, cc] = 255
        use_GPU = torch.cuda.is_available()
        labels, details, _ = segment_with_diameter(
            img,
            model=models.CellposeModel(gpu=use_GPU, model_type="cyto3"),
            diameter=diameter,
            z_axis=z_depth,
            channels=(1, 0),
        )
        # 100 should return empty labels
        if diameter == 100:
            assert np.all(labels == 0), (
                f"Labels should be empty for diameter {diameter}"
            )
        else:
            assert np.any(labels > 0), (
                f"Labels should not be empty for diameter {diameter}"
            )
