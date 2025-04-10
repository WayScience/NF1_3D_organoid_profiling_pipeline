import logging
import pathlib

import numpy
import skimage.io
import skimage.measure

logging.basicConfig(level=logging.INFO)


class ImageSetLoader:
    """
    A class to load an image set consisting of raw z stack images from multiple spectral
    channels and segmentation masks.
    The images are loaded into a dictionary, and various attributes and compartments
    are extracted from the images. The class also provides methods to retrieve images
    and their attributes.
    Parameters
    ----------
    image_set_path : pathlib.Path
        Path to the image set directory.
    anisotropy_spacing : tuple
        The anisotropy spacing of the images. In the format (z_spacing, y_spacing, x_spacing).
    channel_mapping : dict
        A dictionary mapping channel names to their corresponding image file names.
        Example: {'nuclei': 'nuclei_', 'cell': 'cell_', 'cytoplasm': 'cytoplasm_'}
    Attributes
    ----------
    image_set_name : str
        The name of the image set.
    anisotropy_spacing : tuple
        The anisotropy spacing of the images.
    anisotropy_factor : float
        The anisotropy factor calculated from the spacing.
    image_set_dict : dict
        A dictionary containing the loaded images, with keys as channel names.
    unique_mask_objects : dict
        A dictionary containing unique object IDs for each mask in the image set.
    unique_compartment_objects : dict
        A dictionary containing unique object IDs for each compartment in the image set.
        Where a compartment is defined as a segmented region in the image.
        For example typically the Cell, Cytoplasm, Nuclei, and in this case also Organoid.
        The compartments are bounds for measurements.
    image_names : list
        A list of image names in the image set.
    compartments : list
        A list of compartment names in the image set.
    Methods
    -------
    retrieve_image_attributes()
        Retrieves unique object IDs for each mask in the image set.
    get_unique_objects_in_compartments()
        Retrieves unique object IDs for each compartment in the image set.
    get_image(key)
        Retrieves the image corresponding to the specified key from the image set dictionary.
    get_image_names()
        Retrieves the names of images in the image set.
    get_compartments()
        Retrieves the names of compartments in the image set.
    get_anisotropy()
        Retrieves the anisotropy factor.
    """

    def __init__(
        self,
        image_set_path: pathlib.Path,
        anisotropy_spacing: tuple,
        channel_mapping: dict,
    ):
        """
        Initialize the ImageSetLoader with the path to the image set, spacing, and channel mapping.

        Parameters
        ----------
        image_set_path : pathlib.Path
            Path to the image set directory.
        anisotropy_spacing : tuple
            The anisotropy spacing of the images. In the format (z_spacing, y_spacing, x_spacing).
        channel_mapping : dict
            A dictionary mapping channel names to their corresponding image file names.
            Example: {'nuclei': 'nuclei_', 'cell': 'cell_', 'cytoplasm': 'cytoplasm_'}
        """
        self.anisotropy_spacing = anisotropy_spacing
        self.anisotropy_factor = self.anisotropy_spacing[0] / self.anisotropy_spacing[1]
        self.image_set_name = image_set_path.name
        files = sorted(image_set_path.glob("*"))
        files = [f for f in files if f.suffix in [".tif", ".tiff"]]

        # Load images into a dictionary
        self.image_set_dict = {}
        for f in files:
            for key, value in channel_mapping.items():
                if value in f.name:
                    self.image_set_dict[key] = skimage.io.imread(f)

        self.retrieve_image_attributes()
        self.get_compartments()
        self.get_image_names()
        self.get_unique_objects_in_compartments()

    def retrieve_image_attributes(self):
        """
        This is also a quick and dirty way of loading two types of images:
            1. masks (multi-indexed segmentation masks)
            2. The spectral images to extract morphology features from

        My naming convention puts the work "mask" in the segmentation images this
        this is a way to differentiate each mask of each compartment
        apart from the spectral images.

        Future work should be to load the images in a more structured way
        that does not depend on the file naming convention.
        """
        self.unique_mask_objects = {}
        for key, value in self.image_set_dict.items():
            if "mask" in key:
                self.unique_mask_objects[key] = numpy.unique(value)

    def get_unique_objects_in_compartments(self):
        self.unique_compartment_objects = {}
        for compartment in self.compartments:
            self.unique_compartment_objects[compartment] = numpy.unique(
                self.image_set_dict[compartment]
            )
            # remove the 0 label
            self.unique_compartment_objects[compartment] = [
                x for x in self.unique_compartment_objects[compartment] if x != 0
            ]

    def get_image(self, key):
        return self.image_set_dict[key]

    def get_image_names(self):
        self.image_names = [
            x for x in self.image_set_dict.keys() if x not in self.compartments
        ]

    def get_compartments(self):
        self.compartments = [
            x
            for x in self.image_set_dict.keys()
            if "Nuclei" in x or "Cell" in x or "Cytoplasm" in x or "Organoid" in x
        ]

    def get_anisotropy(self):
        return self.anisotropy_spacing[0] / self.anisotropy_spacing[1]


class ObjectLoader:
    """
    A class to load objects from a labeled image and extract their properties.
    Where an object is defined as a segmented region in the image.
    This could be a cell, a nucleus, or any other compartment segmented.
    Parameters
    ----------
    image : numpy.ndarray
        The image from which to extract objects. Preferably a 3D image -> z, y, x
    label_image : numpy.ndarray
        The labeled image containing the segmented objects.
    channel_name : str
        The name of the channel from which the objects are extracted.
    compartment_name : str
        The name of the compartment from which the objects are extracted.
    Attributes
    ----------
    image : numpy.ndarray
        The image from which the objects are extracted.
    label_image : numpy.ndarray
        The labeled image containing the segmented objects.
    channel : str
        The name of the channel from which the objects are extracted.
    compartment : str
        The name of the compartment from which the objects are extracted.
    objects : numpy.ndarray
        The labeled image containing the segmented objects.
    object_ids : numpy.ndarray
        The unique object IDs for the segmented objects.
    Methods
    -------
    __init__(image, label_image, channel_name, compartment_name)
        Initializes the ObjectLoader with the image, label image, channel name, and compartment name.

    """

    def __init__(self, image, label_image, channel_name, compartment_name):
        self.image = image
        self.label_image = label_image
        self.channel = channel_name
        self.compartment = compartment_name
        self.objects = skimage.measure.label(label_image)
        self.object_ids = numpy.unique(self.objects)
        # drop the 0 label
        self.object_ids = self.object_ids[1:]


class TwoObjectLoader:
    """
    A class to load two images and a label image for a specific compartment.
    This class is primarily used for loading images for two-channel analysis like co-localization.
    Parameters
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    compartment : str
        The name of the compartment for which the label image is loaded.
    channel1 : str
        The name of the first channel to be loaded.
    channel2 : str
        The name of the second channel to be loaded.
    Attributes
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    compartment : str
        The name of the compartment for which the label image is loaded.
    label_image : numpy.ndarray
        The labeled image containing the segmented objects for the specified compartment.
    image1 : numpy.ndarray
        The image corresponding to the first channel.
    image2 : numpy.ndarray
        The image corresponding to the second channel.
    object_ids : numpy.ndarray
        The unique object IDs for the segmented objects in the specified compartment.
    Methods
    -------
    __init__(image_set_loader, compartment, channel1, channel2)
        Initializes the TwoObjectLoader with the image set loader, compartment, and channel names.
    """

    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        compartment: str,
        channel1: str,
        channel2: str,
    ):
        self.image_set_loader = image_set_loader
        self.compartment = compartment
        self.label_image = self.image_set_loader.image_set_dict[compartment].copy()
        self.image1 = self.image_set_loader.image_set_dict[channel1].copy()
        self.image2 = self.image_set_loader.image_set_dict[channel2].copy()
        self.object_ids = image_set_loader.unique_compartment_objects[compartment]
