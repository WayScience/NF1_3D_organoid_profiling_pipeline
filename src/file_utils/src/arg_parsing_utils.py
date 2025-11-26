import argparse


def check_for_missing_args(**kwargs):
    """
    Check if any required arguments are missing.

    Raises
    ------
    ValueError
        If any required arguments are missing.
    """
    missing_args = []
    for arg, value in kwargs.items():
        if value is None:
            missing_args.append(arg)
    if missing_args:
        raise ValueError(
            f"Missing required arguments: {', '.join(missing_args)}. "
            "Please provide all required arguments."
        )


def parse_args():
    """
    Parse command line arguments for segmentation tasks.

    Returns
    -------
    dict
        A dictionary containing the parsed arguments.
        Where:
            'well_fov' is the well and field of view to process,
            'patient' is the patient ID, 'window_size' is the window size for image processing,
            'clip_limit' is the clip limit for contrast enhancement, and
            'compartment' is the compartment to process.
    Raises
    ------
    ValueError
        If any required arguments are missing.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default=None,
        help="Well and field of view to process, e.g. 'A01-1'",
    )
    argparser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Patient ID, e.g. 'NF0014'",
    )
    argparser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for image processing, e.g. 3",
    )
    argparser.add_argument(
        "--clip_limit",
        type=float,
        default=None,
        help="Clip limit for contrast enhancement, e.g. 0.05",
    )
    argparser.add_argument(
        "--compartment",
        type=str,
        default=None,
        help="Compartment to process, e.g. 'Nuclei'",
    )
    argparser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Channel to process, e.g. 'DAPI'",
    )
    argparser.add_argument(
        "--processor_type",
        type=str,
        default=None,
        help="Type of processor to use, e.g. 'CPU' or 'GPU'",
    )
    argparser.add_argument(
        "--input_subparent_name",
        type=str,
        default=None,
        help="Name of the subparent directory for input images, e.g. 'deconvolved_images'",
    )
    argparser.add_argument(
        "--mask_subparent_name",
        type=str,
        default=None,
        help="Name of the subparent directory for segmentation masks, e.g. 'deconvolved_segmentation_masks'",
    )
    argparser.add_argument(
        "--output_features_subparent_name",
        type=str,
        default=None,
        help="Name of the subparent directory for output features, e.g. 'sammed3D_features'",
    )
    argparser.add_argument(
        "--converted_profiles_intermediate_subparent_name",
        type=str,
        default=None,
        help="Name of the subparent directory for converted profiles intermediate files, e.g. 'converted_profiles_intermediate'",
    )
    argparser.add_argument(
        "--image_based_profiles_subparent_name",
        type=str,
        default=None,
        help="Name of the subparent directory for image based profiles, e.g. 'image_based_profiles'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
    window_size = args.window_size
    clip_limit = args.clip_limit
    compartment = args.compartment
    channel = args.channel
    processor_type = args.processor_type
    input_subparent_name = args.input_subparent_name
    mask_subparent_name = args.mask_subparent_name
    output_features_subparent_name = args.output_features_subparent_name
    image_based_profiles_subparent_name = args.image_based_profiles_subparent_name

    return {
        "well_fov": well_fov,
        "patient": patient,
        "window_size": window_size,
        "clip_limit": clip_limit,
        "compartment": compartment,
        "channel": channel,
        "processor_type": processor_type,
        "input_subparent_name": input_subparent_name,
        "mask_subparent_name": mask_subparent_name,
        "output_features_subparent_name": output_features_subparent_name,
        "image_based_profiles_subparent_name": image_based_profiles_subparent_name,
    }
