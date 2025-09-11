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

    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
    window_size = args.window_size
    clip_limit = args.clip_limit
    compartment = args.compartment
    channel = args.channel
    processor_type = args.processor_type

    return {
        "well_fov": well_fov,
        "patient": patient,
        "window_size": window_size,
        "clip_limit": clip_limit,
        "compartment": compartment,
        "channel": channel,
        "processor_type": processor_type,
    }
