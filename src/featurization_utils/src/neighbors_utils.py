from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy
import pandas
import skimage.measure
from loading_classes import ObjectLoader
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis


def neighbors_expand_box(
    min_coor: Union[int, float],
    max_coord: Union[int, float],
    current_min: Union[int, float],
    current_max: Union[int, float],
    expand_by: int,
) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Expand the bounding box of the object by a specified distance in each direction.

    Parameters
    ----------
    min_coor : Union[int, float]
        The global minimum coordinate of the image.
    max_coord : Union[int, float]
        The global maximum coordinate of the image.
    current_min : Union[int, float]
        The current minimum coordinate of the object.
    current_max : Union[int, float]
        The current maximum coordinate of the object.
    expand_by : int
        The distance by which to expand the bounding box.

    Returns
    -------
    Tuple[Union[int, float], Union[int, float]]
        The new minimum and maximum coordinates of the bounding box.
    """
    if current_min - expand_by < min_coor:
        current_min = min_coor
    else:
        current_min -= expand_by
    if current_max + expand_by > max_coord:
        current_max = max_coord
    else:
        current_max += expand_by
    return current_min, current_max


# crop the image to the bbox of the mask
def crop_3D_image(
    image: numpy.ndarray,
    bbox: Tuple[
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
    ],
) -> numpy.ndarray:
    """
    Crop the 3D image to the bounding box of the object.

    Parameters
    ----------
    image : numpy.ndarray
        The 3D image to be cropped.
    bbox : Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]]
        The bounding box of the object in the format (z1, y1, x1, z2, y2, x2).

    Returns
    -------
    numpy.ndarray
        The cropped 3D image.
    """
    z1, y1, x1, z2, y2, x2 = bbox
    return image[z1:z2, y1:y2, x1:x2]


def measure_3D_number_of_neighbors(
    object_loader: ObjectLoader,
    distance_threshold: int = 10,
    anisotropy_factor: int = 10,
) -> Dict[str, list]:
    """
    This function calculates the number of neighbors for each object in a 3D image.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader object that contains the image and label image.
    distance_threshold : int, optional
        The distance threshold for counting neighbors, by default 10
    anisotropy_factor : int, optional
        The anisotropy factor for the image where the anisotropy factor is the ratio of the pixel size in the z direction to the pixel size in the x and y directions, by default 10

    Returns
    -------
    Dict[str, list]
        A dictionary containing the object ID and the number of neighbors for each object.
    """
    label_object = object_loader.label_image
    labels = object_loader.object_ids
    # set image global min and max coordinates
    image_global_min_coord_z = 0
    image_global_min_coord_y = 0
    image_global_min_coord_x = 0
    image_global_max_coord_z = label_object.shape[0]
    image_global_max_coord_y = label_object.shape[1]
    image_global_max_coord_x = label_object.shape[2]

    neighbors_out_dict = {
        "object_id": [],
        "Neighbors_adjacent": [],
        f"Neighbors_{distance_threshold}": [],
    }
    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_label_object[selected_label_object != label] = 0
        props_label = skimage.measure.regionprops_table(
            selected_label_object, properties=["bbox"]
        )
        # get the number of neighbors for each object
        distance_x_y = distance_threshold
        distance_z = numpy.ceil(distance_threshold / anisotropy_factor).astype(int)
        # find how many other indexes are within a specified distance of the object
        # first expand the mask image by a specified distance
        z_min, y_min, x_min, z_max, y_max, x_max = (
            props_label["bbox-0"][0],
            props_label["bbox-1"][0],
            props_label["bbox-2"][0],
            props_label["bbox-3"][0],
            props_label["bbox-4"][0],
            props_label["bbox-5"][0],
        )
        original_bbox = (z_min, y_min, x_min, z_max, y_max, x_max)

        new_z_min, new_z_max = neighbors_expand_box(
            min_coor=image_global_min_coord_z,
            max_coord=image_global_max_coord_z,
            current_min=z_min,
            current_max=z_max,
            expand_by=distance_z,
        )
        new_y_min, new_y_max = neighbors_expand_box(
            min_coor=image_global_min_coord_y,
            max_coord=image_global_max_coord_y,
            current_min=y_min,
            current_max=y_max,
            expand_by=distance_x_y,
        )
        new_x_min, new_x_max = neighbors_expand_box(
            min_coor=image_global_min_coord_x,
            max_coord=image_global_max_coord_x,
            current_min=x_min,
            current_max=x_max,
            expand_by=distance_x_y,
        )
        bbox = (new_z_min, new_y_min, new_x_min, new_z_max, new_y_max, new_x_max)
        croppped_neighbor_image = crop_3D_image(image=label_object, bbox=bbox)
        self_cropped_neighbor_image = crop_3D_image(
            image=label_object, bbox=original_bbox
        )
        # find all the unique values in the cropped image of the object of interest
        # this is the number of neighbors in the cropped image
        n_neighbors_adjacent = (
            len(
                numpy.unique(
                    self_cropped_neighbor_image[self_cropped_neighbor_image > 0]
                )
            )
            - 1
        )

        # find all the unique values in the expanded cropped image of the object of interest
        # this gives the number of neighbors in a n distance of the object
        n_neighbors_by_distance = (
            len(numpy.unique(croppped_neighbor_image[croppped_neighbor_image > 0])) - 1
        )
        neighbors_out_dict["object_id"].append(label)
        neighbors_out_dict["Neighbors_adjacent"].append(n_neighbors_adjacent)
        neighbors_out_dict[f"Neighbors_{distance_threshold}"].append(
            n_neighbors_by_distance
        )

    return neighbors_out_dict


def get_coordinates(nuclei_mask: numpy.ndarray, object_ids=[]) -> pandas.DataFrame:
    """
    Extract coordinates from a labeled mask.

    Parameters:
    -----------
    nuclei_mask : ndarray
        3D labeled mask where each object has a unique ID
    object_ids : list
        List of object IDs to extract

    Returns:
    --------
    coords : pandas.DataFrame
        DataFrame with columns: object_id, x, y, z
    """
    coords = {"object_id": [], "x": [], "y": [], "z": []}

    for obj_id in object_ids:
        object_mask = nuclei_mask.copy()
        object_mask[object_mask != obj_id] = 0
        object_mask[object_mask == obj_id] = 1
        # Get the centroid of the object
        z, y, x = numpy.where(object_mask == 1)
        centroid = (numpy.mean(x), numpy.mean(y), numpy.mean(z))
        coords["object_id"].append(obj_id)
        coords["x"].append(centroid[0])
        coords["y"].append(centroid[1])
        coords["z"].append(centroid[2])

    return pandas.DataFrame(coords)


def calculate_centroid(coords: pandas.DataFrame) -> numpy.ndarray:
    """Calculate the centroid of cell coordinates."""
    return numpy.mean(coords, axis=0)


def euclidean_distance_from_centroid(
    coords: numpy.ndarray, centroid: numpy.ndarray
) -> numpy.ndarray:
    """Calculate Euclidean distance from centroid for each cell."""
    return numpy.sqrt(numpy.sum((coords - centroid) ** 2, axis=1))


def mahalanobis_distance_from_centroid(
    coords: numpy.ndarray, centroid: numpy.ndarray, min_cells_threshold: int = 50
) -> numpy.ndarray:
    """
    Calculate Mahalanobis distance from centroid for each cell.
    This accounts for the covariance structure (shape) of the organoid.

    For small sample sizes (<50 cells), uses regularization or falls back to Euclidean.

    Parameters:
    -----------
    coords : ndarray
        Cell coordinates (n_cells, 3)
    centroid : ndarray
        Centroid coordinates (3,)
    min_cells_threshold : int
        Minimum cells needed for reliable Mahalanobis (default: 50)

    Returns:
    --------
    distances : ndarray
        Mahalanobis distances for each cell
    """
    n_cells = len(coords)

    # For very small samples, use Euclidean distance instead
    if n_cells < 20:
        print(
            f"  WARNING: Only {n_cells} cells. Using Euclidean distance instead of Mahalanobis."
        )
        return euclidean_distance_from_centroid(coords, centroid)

    # Calculate covariance matrix
    cov_matrix = numpy.cov(coords.T)

    # For small samples (20-50), use strong regularization
    if n_cells < min_cells_threshold:
        # Regularization strength inversely proportional to sample size
        reg_strength = (min_cells_threshold - n_cells) / min_cells_threshold * 0.1
        cov_matrix += numpy.eye(3) * reg_strength * numpy.trace(cov_matrix) / 3
        print(
            f"  WARNING: Only {n_cells} cells. Using regularized covariance (λ={reg_strength:.3f})"
        )
    else:
        # Standard small regularization for numerical stability
        cov_matrix += numpy.eye(3) * 1e-6

    # Calculate inverse covariance matrix
    try:
        inv_cov = numpy.linalg.inv(cov_matrix)
    except numpy.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        print("  WARNING: Singular covariance matrix. Using pseudo-inverse.")
        inv_cov = numpy.linalg.pinv(cov_matrix)

    # Calculate Mahalanobis distance for each point
    distances = numpy.array(
        [
            numpy.sqrt((coord - centroid).T @ inv_cov @ (coord - centroid))
            for coord in coords
        ]
    )

    return distances


def classify_cells_into_shells(
    coords: pandas.DataFrame or dict,
    n_shells: int = 5,
    method: str = "mahalanobis",
    min_cells_per_shell: int = 3,
) -> dict:
    """
    Classify cells into radial shells based on distance from centroid.

    Automatically adjusts n_shells for small organoids to ensure meaningful statistics.

    Parameters:
    -----------
    coords : pandas.DataFrame or dict
        Cell coordinates with columns/keys: object_id, x, y, z
    n_shells : int
        Number of concentric shells to create (will be adjusted if needed)
    method : str
        'euclidean' or 'mahalanobis'
    min_cells_per_shell : int
        Minimum average cells per shell (default: 3)

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'shell_assignments': Shell number for each cell (0 = innermost)
        - 'distances_from_center': Distance from centroid for each cell
        - 'distances_from_exterior': Distance from exterior for each cell
        - 'normalized_distances_from_center': Normalized distances (0-1)
        - 'centroid': Centroid coordinates
        - 'max_distance': Maximum distance from centroid
        - 'n_shells_used': Actual number of shells used
    """
    # Handle both DataFrame and dict input
    if isinstance(coords, pandas.DataFrame):
        object_ids = coords["object_id"].to_numpy()
        coords_array = coords[["x", "y", "z"]].to_numpy()
    else:
        object_ids = numpy.array(coords["object_id"])
        coords_array = numpy.column_stack([coords["x"], coords["y"], coords["z"]])
    if len(coords_array) == 0:
        results = {
            "object_id": [],
            "shell_assignments": [],
            "distances_from_center": [],
            "distances_from_exterior": [],
            "normalized_distances_from_center": [],
        }
        centroid = None
        return results, centroid
    n_cells = len(coords_array)
    centroid = calculate_centroid(coords_array)

    # Adjust number of shells for small organoids
    max_shells = max(2, n_cells // min_cells_per_shell)
    if n_shells > max_shells:
        print(
            f"  WARNING: {n_cells} cells with {n_shells} shells = {n_cells / n_shells:.1f} cells/shell"
        )
        print(f"           Reducing to {max_shells} shells for statistical reliability")
        n_shells = max_shells

    # Calculate distances based on method
    if method == "mahalanobis":
        distances = mahalanobis_distance_from_centroid(coords_array, centroid)
    else:  # euclidean
        distances = euclidean_distance_from_centroid(coords_array, centroid)

    # Normalize distances to 0-1 range
    max_distance = numpy.percentile(
        distances, 95
    )  # Use 95 percentile to avoid outliers
    # max_distance = numpy.max(distances)
    normalized_distances = distances / max_distance

    # Assign shells (0 = innermost, n_shells-1 = outermost)
    shell_assignments = numpy.floor(normalized_distances * n_shells).astype(int)
    shell_assignments = numpy.clip(shell_assignments, 0, n_shells - 1)

    # Calculate distance from exterior (inverse of distance from center)
    distance_from_exterior = max_distance - distances

    results = {
        "object_id": object_ids,
        "shell_assignments": shell_assignments,
        "distances_from_center": distances,
        "distances_from_exterior": distance_from_exterior,
        "normalized_distances_from_center": normalized_distances,
    }

    return results, centroid


def create_results_dataframe(results: dict) -> pandas.DataFrame:
    """
    Create a pandas DataFrame with all cell information.

    Parameters:
    -----------
    results : dict
        Results from classify_cells_into_shells

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with cell information
    """
    # Handle both DataFrame and dict input
    if isinstance(results, dict):
        df = pandas.DataFrame.from_dict(results)
    else:
        raise ValueError(
            "Input must be a results dictionary from classify_cells_into_shells."
        )

    return df


def visualize_organoid_shells(
    coords: pandas.DataFrame,
    classification_results: dict,
    title: str = "Organoid Shell Classification",
    centroid: numpy.ndarray = None,
) -> plt.figure:
    """
    Create 3D visualization of organoid with shell coloring.

    Parameters:
    -----------
    coords : pandas.DataFrame or dict
        Cell coordinates with columns/keys: object_id, x, y, z
    classification_results : dict
        Results from classify_cells_into_shells
    title : str
        Plot title
    """
    # Handle both DataFrame and dict input
    if isinstance(coords, pandas.DataFrame):
        x_coords = coords["x"].to_numpy()
        y_coords = coords["y"].to_numpy()
        z_coords = coords["z"].to_numpy()
    else:
        x_coords = numpy.array(coords["x"])
        y_coords = numpy.array(coords["y"])
        z_coords = numpy.array(coords["z"])

    fig = plt.figure(figsize=(14, 6))

    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection="3d")

    shell_assignments = classification_results["shell_assignments"]
    n_shells = classification_results.get(
        "n_shells_used", len(numpy.unique(shell_assignments))
    )

    # Red to blue color gradient
    colors = plt.cm.RdYlBu_r(numpy.linspace(0, 1, n_shells))

    for shell in range(n_shells):
        mask = shell_assignments == shell
        if numpy.sum(mask) > 0:  # Only plot if shell has cells
            ax1.scatter(
                x_coords[mask],
                y_coords[mask],
                z_coords[mask],
                c=[colors[shell]],
                label=f"Shell {shell + 1} (n={numpy.sum(mask)})",
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

    if centroid is not None:
        ax1.scatter(
            *centroid,
            c="black",
            s=200,
            marker="*",
            label="Centroid",
            edgecolors="white",
            linewidths=2,
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8)

    # Shell distribution histogram
    ax2 = fig.add_subplot(122)
    shell_counts = [numpy.sum(shell_assignments == i) for i in range(n_shells)]
    bars = ax2.bar(
        range(1, n_shells + 1), shell_counts, color=colors, alpha=0.7, edgecolor="black"
    )
    ax2.set_xlabel("Shell Number")
    ax2.set_ylabel("Number of Cells")
    ax2.set_title("Cell Distribution Across Shells")
    ax2.set_xticks(range(1, n_shells + 1))

    # Add percentage labels on bars
    total_cells = len(x_coords)
    for i, (bar, count) in enumerate(zip(bars, shell_counts)):
        height = bar.get_height()
        percentage = (count / total_cells) * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add horizontal line for average
    avg_per_shell = total_cells / n_shells
    ax2.axhline(
        y=avg_per_shell,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Average ({avg_per_shell:.1f})",
    )
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_distance_distributions(
    classification_results: dict, n_shells: int = None
) -> plt.figure:
    """
    Plot distance distributions for each shell.

    Parameters:
    -----------
    classification_results : dict
        Results from classify_cells_into_shells
    n_shells : int, optional
        Number of shells (will use n_shells_used from results if not provided)
    """
    if n_shells is None:
        n_shells = classification_results.get(
            "n_shells_used",
            len(numpy.unique(classification_results["shell_assignments"])),
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    shell_assignments = classification_results["shell_assignments"]
    distances_from_center = classification_results["distances_from_center"]
    distances_from_exterior = classification_results["distances_from_exterior"]

    colors = plt.cm.RdYlBu_r(numpy.linspace(0, 1, n_shells))

    # Distance from center
    for shell in range(n_shells):
        mask = shell_assignments == shell
        if numpy.sum(mask) > 0:
            axes[0].hist(
                distances_from_center[mask],
                bins=20,
                alpha=0.5,
                color=colors[shell],
                label=f"Shell {shell + 1}",
                edgecolor="black",
            )

    axes[0].set_xlabel("Distance from Center")
    axes[0].set_ylabel("Number of Cells")
    axes[0].set_title("Distance from Center Distribution")
    axes[0].legend()

    # Distance from exterior
    for shell in range(n_shells):
        mask = shell_assignments == shell
        if numpy.sum(mask) > 0:
            axes[1].hist(
                distances_from_exterior[mask],
                bins=20,
                alpha=0.5,
                color=colors[shell],
                label=f"Shell {shell + 1}",
                edgecolor="black",
            )

    axes[1].set_xlabel("Distance from Exterior")
    axes[1].set_ylabel("Number of Cells")
    axes[1].set_title("Distance from Exterior Distribution")
    axes[1].legend()

    plt.tight_layout()
    return fig
