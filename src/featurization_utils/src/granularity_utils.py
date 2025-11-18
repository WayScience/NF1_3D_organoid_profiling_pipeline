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
from skimage.draw import ellipsoid


def granularity_feature(length):
    C_GRANULARITY = "GRANULARITY.%s"
    return C_GRANULARITY % (length)


def measure_3D_granularity(
    object_loader: ObjectLoader,
    radius: int = 1,
    granular_spectrum_length: int = 16,
) -> Dict[str, float]:
    """
    This function calculates the granularity of an image using the
    granularity feature. It uses the skimage library to perform the calculations.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader that contains the image and label image.
    radius : int, optional
        The radius of the ball used for morphological operations, by default 1
    granular_spectrum_length : int, optional
        The length of the granular spectrum, by default 16

    Returns
    -------
    Dict[str, float]
        A dictionary containing the granularity feature measurements per object ID.
    """
    # Transcribed from the Matlab module: granspectr function
    #
    # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
    # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
    # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
    # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
    # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
    # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
    # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
    # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
    #
    # further adapted from the transcribed code in the CellProfiler source code
    # no image-level granularity is calculated, only object-level granularity
    footprint = skimage.morphology.ball(radius, dtype=bool)

    object_measurements = {"object_id": [], "feature": [], "value": []}

    for label in tqdm.tqdm(
        object_loader.object_ids, desc="Processing objects", position=0, leave=True
    ):
        mask = object_loader.label_image.copy()
        pixels = object_loader.image.copy()

        mask[mask != label] = 0
        pixels[mask != label] = 0
        mask[mask == label] = 1  # binarize the mask
        mask = mask.astype(bool)  # convert to boolean mask

        startmean = numpy.mean(pixels[mask])
        ero = pixels.copy()

        # Mask the test image so that masked pixels will have no effect
        # during reconstruction
        ero[~mask] = 0
        currentmean = startmean
        startmean = max(startmean, numpy.finfo(float).eps)

        for i in tqdm.tqdm(
            range(1, granular_spectrum_length + 1),
            desc=f"Granularity for label {label}",
            position=1,
            leave=False,
        ):
            prevmean = currentmean
            # this is an iterative process thus if the prevmean is 0 we
            # should stop iterating as the signal has been diminished
            # all further iterations will be 0 no need to perform
            # expensive erosion and reconstruction iterations on blank images
            if prevmean != 0:
                ero_mask = numpy.zeros_like(ero)
                ero_mask[mask == True] = ero[mask == True]
                ero = skimage.morphology.erosion(ero_mask, footprint=footprint)

                rec = skimage.morphology.reconstruction(
                    ero, pixels, footprint=footprint
                )
                currentmean = numpy.mean(rec[mask])
                gs = (prevmean - currentmean) * 100 / startmean
                feature = granularity_feature(i)

                new_mean = scipy.ndimage.mean(rec)
                gss = (currentmean - new_mean) * 100 / startmean
                currentmean = new_mean
            else:
                gss = 0.0
                feature = granularity_feature(i)

            object_measurements["object_id"].append(label)
            object_measurements["feature"].append(feature)
            object_measurements["value"].append(gss)
    return object_measurements
