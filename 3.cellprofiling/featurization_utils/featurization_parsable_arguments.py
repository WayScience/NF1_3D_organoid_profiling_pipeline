import argparse

from errors import ProcessorTypeError


def check_for_missing_args(**kwargs):
    missing_args = []
    for arg, value in kwargs.items():
        if value is None:
            missing_args.append(arg)
    if missing_args:
        raise ValueError(
            f"Missing required arguments: {', '.join(missing_args)}. "
            "Please provide all required arguments."
        )


def parse_featurization_args():
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
        "--channel",
        type=str,
        default=None,
        help="Channel to process, e.g. 'DNA'",
    )
    argparser.add_argument(
        "--compartment",
        type=str,
        default=None,
        help="Compartment to process, e.g. 'Nuclei'",
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
    channel = args.channel
    compartment = args.compartment
    processor_type = args.processor_type
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        channel=channel,
        compartment=compartment,
        processor_type=processor_type,
    )
    if processor_type not in ["CPU", "GPU"]:
        raise ProcessorTypeError("Processor type not recognized. Use 'CPU' or 'GPU'.")
    return {
        "well_fov": well_fov,
        "patient": patient,
        "channel": channel,
        "compartment": compartment,
        "processor_type": processor_type,
    }
