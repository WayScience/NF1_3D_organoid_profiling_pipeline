from typing import Dict

import cucim
import cucim.skimage.morphology
import cupy
import cupyx
import cupyx.scipy.ndimage
import numpy
import pandas
import scipy
import skimage
import tqdm
from loading_classes import ImageSetLoader, ObjectLoader


def granularity_feature(length):
    C_GRANULARITY = "GRANULARITY.%s"
    return C_GRANULARITY % (length)


class ObjectGranularityRecord:
    """
    This class is used to store the information about the object.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader that contains the image and label image.
    object_index : int
        The index of the object to be measured.
    Attributes
    ----------
    object_index : int
        The index of the object to be measured.
    labels : numpy.ndarray
        The labels of the object in the image.
    image : numpy.ndarray
        The image of the object.
    nobjects : int
        The number of objects in the image.
    range : numpy.ndarray
        The range of the object labels.
    current_mean : float
        The current mean intensity of the object.
    start_mean : float
        The starting mean intensity of the object.
    Methods
    -------
    __init__(object_loader, object_index)
        Initializes the ObjectRecord with the object loader and object index.
    """

    def __init__(self, object_loader, object_index):
        self.object_index = object_index
        self.labels = object_loader.objects.copy()
        # select the object
        self.labels[self.labels != object_index] = 0
        self.image = object_loader.image.copy()
        self.image[self.labels != object_index] = 0

        self.nobjects = len(numpy.unique(self.labels))
        if self.nobjects != 0:
            self.current_mean = scipy.ndimage.mean(self.image, self.labels)

            self.start_mean = numpy.maximum(self.current_mean, numpy.finfo(float).eps)
        else:
            raise ValueError(
                f"No objects found in the image for object index {object_index}"
            )


def measure_3D_granularity(
    object_loader: ObjectLoader,
    radius: int = 10,
    granular_spectrum_length: int = 16,
    subsample_size: float = 0.25,
    image_name: str = "image",
) -> Dict[str, float]:
    """
    This function calculates the granularity of an image using the
    granularity feature. It uses the skimage library to perform the calculations.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader that contains the image and label image.
    radius : int, optional
        The radius of the ball used for morphological operations, by default 10
    granular_spectrum_length : int, optional
        The length of the granular spectrum, by default 16
    subsample_size : float, optional
        The size of the image subsample, by default 0.25
    image_name : str, optional
        The name of the image, by default "image"

    Returns
    -------
    Dict[str, float]
        A dictionary containing the granularity feature measurements per object ID.
    """

    image_object = object_loader.image
    label_object = object_loader.label_image

    pixels = image_object.copy()
    mask = label_object.copy()

    # begin by downsampling the image
    new_shape = numpy.array(pixels.shape)
    new_shape = (
        new_shape * subsample_size
    )  # downsample the image for computational speed
    k, i, j = (
        numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
        / subsample_size
    )
    pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
    mask = (
        scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9
    )  # 0.9 is a threshold to determine if the pixel is part of the object

    back_shape = new_shape
    k, i, j = (
        numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
        / subsample_size
    )
    back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
    back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9

    footprint = skimage.morphology.ball(radius, dtype=bool)

    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)

    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]

    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    k, i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
        float
    )
    k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
    i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
    j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
    back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    object_records = (
        ObjectGranularityRecord(object_loader, object_index)
        for _, object_index in enumerate(object_loader.object_ids)
    )

    startmean = numpy.mean(pixels[mask])
    ero = pixels.copy()

    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, numpy.finfo(float).eps)
    footprint = skimage.morphology.ball(1, dtype=bool)
    feature_measurments = {}
    object_measurements = {"object_id": [], "feature": [], "value": []}
    for i in range(1, granular_spectrum_length + 1):
        prevmean = currentmean
        ero_mask = numpy.zeros_like(ero)
        ero_mask[mask == True] = ero[mask == True]
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)

        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = numpy.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
        feature = granularity_feature(i)
        feature_measurments[feature] = gs
        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        orig_shape = image_object.shape
        k, i, j = numpy.mgrid[
            0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]
        ].astype(float)
        k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
        i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
        j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
        rec = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)
        for object_record in object_records:
            assert isinstance(object_record, ObjectGranularityRecord)
            if object_record.nobjects > 0:
                new_mean = scipy.ndimage.mean(rec, object_record.labels)
                gss = (
                    (object_record.current_mean - new_mean)
                    * 100
                    / object_record.start_mean
                )
                object_record.current_mean = new_mean
            else:
                gss = numpy.zeros((0,))
            object_measurements["object_id"].append(object_record.object_index)
            object_measurements["feature"].append(feature)
            object_measurements["value"].append(gss)
    return object_measurements


def measure_3D_granularity_gpu(
    object_loader: ObjectLoader,
    image_set_loader: ImageSetLoader,
    radius: int = 10,
    granular_spectrum_length: int = 16,
    subsample_size: float = 0.25,
    image_name: str = "image",
) -> dict:
    """
    This function calculates the granularity of an image using the
    granularity feature. It uses the cupy library to perform the calculations
    on the GPU.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader that contains the image and label image.
    image_set_loader : ImageSetLoader
        The image set loader object that contains the image and label image.
    radius : int, optional
        The radius of the ball used for morphological operations, by default 10
    granular_spectrum_length : int, optional
        The length of the granular spectrum, by default 16
    subsample_size : float, optional
        The size of the image subsample, by default 0.25
    image_name : str, optional
        The name of the image, by default "image"

    Returns
    -------
    dict
        A dictionary containing the granularity feature measurements per object ID.
    """

    pixels = cupy.asarray(object_loader.image)
    mask = cupy.asarray(object_loader.label_image)
    new_shape = cupy.asarray(pixels.shape) * subsample_size
    k, i, j = (
        cupy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
            cupy.float32
        )
        / subsample_size
    )
    pixels = cupyx.scipy.ndimage.map_coordinates(pixels, cupy.stack((k, i, j)), order=1)
    mask = (
        cupyx.scipy.ndimage.map_coordinates(
            mask,
            cupy.stack((k, i, j)),
        )
        > 0.9
    )
    back_shape = (
        new_shape * subsample_size
    )  # downsample the image for computational speed

    k, i, j = (
        cupy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
            cupy.float32
        )
        / subsample_size
    )
    back_pixels = cupyx.scipy.ndimage.map_coordinates(
        pixels, cupy.stack((k, i, j)), order=1
    )
    back_mask = (
        cupyx.scipy.ndimage.map_coordinates(
            mask.astype(cupy.float32), cupy.stack((k, i, j))
        )
        > 0.9
    )
    footprint = cucim.skimage.morphology.ball(radius, dtype=bool)

    back_pixels_mask = cupy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels_mask = cupy.asarray(back_pixels_mask)

    # getting a jitify runtime error
    # bad fix and workaround -> I do not like converting between numpy and cupy
    # too much overhead...
    # convert to numpy array
    # run the skimage erosion
    # convert back to cupy array
    # the cucim version of skimage does not work with jitify for erosion or dilation
    back_pixels_mask = back_pixels_mask.get()
    footprint = footprint.get()
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = cupy.asarray(back_pixels_mask)
    back_pixels = cupy.asarray(back_pixels)
    footprint = cupy.asarray(footprint)

    # bad workaround but yet here we are with features

    back_pixels_mask = cupy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]

    # getting a jitify runtime error
    # bad fix and workaround
    # convert to numpy array
    # run the skimage erosion
    # convert back to cupy array
    # the cucim version of skimage does not work with jitify for erosion or dilation

    back_pixels_mask = back_pixels_mask.get()
    footprint = footprint.get()
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    back_pixels_mask = cupy.asarray(back_pixels_mask)
    back_pixels = cupy.asarray(back_pixels)
    footprint = cupy.asarray(footprint)
    # bad workaround but yet here we are with features

    k, i, j = cupy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
        cupy.float32
    )

    k *= (back_shape[0] - 1) / (new_shape[0] - 1)
    i *= (back_shape[1] - 1) / (new_shape[1] - 1)
    j *= (back_shape[2] - 1) / (new_shape[2] - 1)

    back_pixels = cupyx.scipy.ndimage.map_coordinates(
        back_pixels, cupy.stack((k, i, j)), order=1
    )
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    startmean = cupy.mean(pixels[mask])
    ero = pixels.copy()

    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, cupy.finfo(float).eps)
    footprint = cucim.skimage.morphology.ball(1, dtype=bool)
    feature_measurments = {}
    objects_records = (
        ObjectGranularityRecord(object_loader=object_loader, object_index=object_id)
        for object_id in object_loader.object_ids
    )
    object_measurements = {"object_id": [], "feature": [], "value": []}
    for i in tqdm.tqdm(range(1, granular_spectrum_length + 1)):
        prevmean = currentmean
        ero_mask = cupy.zeros_like(ero)
        ero_mask[mask == True] = ero[mask == True]
        # getting a jitify runtime error
        # bad fix and workaround
        # convert to numpy array
        # run the skimage erosion
        # convert back to cupy array
        # the cucim version of skimage does not work with jitify for erosion or dilation

        ero_mask = ero_mask.get()
        footprint = footprint.get()
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        ero = cupy.asarray(ero)
        ero_mask = cupy.asarray(ero_mask)
        footprint = cupy.asarray(footprint)
        # bad workaround but yet here we are with features

        rec = cucim.skimage.morphology.reconstruction(
            ero,
            pixels,
            footprint=footprint,
        )
        currentmean = cupy.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
        feature = granularity_feature(i)
        feature_measurments[feature] = gs
        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        orig_shape = object_loader.image.shape
        k, i, j = cupy.mgrid[
            0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]
        ].astype(cupy.float32)
        k *= (new_shape[0] - 1) / (orig_shape[0] - 1)
        i *= (new_shape[1] - 1) / (orig_shape[1] - 1)
        j *= (new_shape[2] - 1) / (orig_shape[2] - 1)

        rec = cupyx.scipy.ndimage.map_coordinates(rec, cupy.stack((k, i, j)), order=1)
        for object_record in objects_records:
            if object_record.nobjects > 0:
                labels = cupy.asarray(object_record.labels)
                new_mean = cupy.mean(rec - labels)
                gss = (
                    (object_record.current_mean - new_mean)
                    * 100
                    / object_record.start_mean
                )
                object_record.current_mean = new_mean

            else:
                gss = cupy.zeros((0,))
            object_measurements["object_id"].append(object_record.object_index)
            object_measurements["feature"].append(feature)
            object_measurements["value"].append(gss.get())
    return object_measurements
