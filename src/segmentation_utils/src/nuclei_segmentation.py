import cellpose.models as models
import torch
import tqdm


# ----------------------------------------------------------------------
# cellpose segementation
# ----------------------------------------------------------------------
def segmentaion_on_two_D(imgs):
    use_GPU = torch.cuda.is_available()
    # Load the model

    model = models.CellposeModel(
        gpu=use_GPU,
    )
    output_dict = {
        "slice": [],
        "labels": [],
        "details": [],
    }
    for slice in tqdm.tqdm(range(imgs.shape[0])):
        # Perform segmentation
        output_dict["slice"].append(slice)
        labels, details, _ = model.eval(
            imgs[slice, :, :],
        )
        output_dict["labels"].append(labels)
        output_dict["details"].append(details)
    return output_dict
