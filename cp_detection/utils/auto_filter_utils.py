import os
import sys
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.image_utils import watershed, remove_small_objects, detect_circles_and_semi_circles, merge_and_segment, remove_duplicates
from utils.json_utils import json2list, list2json, save_filtered_json
import json

def auto_filter(json_input: str, min_circular_ratio: float = 0.9, max_circular_ratio: float = 1.05,
                min_semi_ratio: float = 0.45, max_semi_ratio: float = 0.55, min_size_ratio: float = 0.005,
                second_min_circular_ratio: float = 0.7) -> List[np.ndarray]:
    """
    Apply an automated filter to the annotations in a JSON string.

    Args:
        json_input (str): A JSON string of annotations.
        min_circular_ratio (float): The minimum area ratio for a region to be considered a circle.
        max_circular_ratio (float): The maximum area ratio for a region to be considered a circle.
        min_semi_ratio (float): The minimum area ratio for a region to be considered a semi-circle.
        max_semi_ratio (float): The maximum area ratio for a region to be considered a semi-circle.
        min_size_ratio (float): The minimum object size as a ratio of the total image size.
        second_min_circular_ratio (float): The minimum area ratio for a region to be considered a circle in the second pass.

    Returns:
        list_of_arrays (List[np.ndarray]): A list of numpy arrays each representing a unique mask after filtering.
    """
    # Convert the JSON string into a list of numpy arrays
    arrays = json2list(json.loads(json_input))

    # Initialize a list to store the candidates
    candidates = [[], []]

    # For each array
    for array in tqdm(arrays, total = len(arrays), file = sys.stdout):
        # Perform watershed segmentation
        labels = watershed(array)

        # Remove small objects
        labels = remove_small_objects(labels, min_size_ratio)

        # Detect circles and semi-circles
        circle_regions, semi_circle_regions = detect_circles_and_semi_circles(
            labels, min_circular_ratio, max_circular_ratio, min_semi_ratio, max_semi_ratio
        )

        # Add the circle regions to the list of candidates
        candidates[0].extend(circle_regions)

        # Add the semi-circle regions to the list of candidates
        candidates[1].extend(semi_circle_regions)

    if candidates[1]:
        # Merge and segment the semi-circle candidates
        masks = merge_and_segment(candidates[1], second_min_circular_ratio, max_circular_ratio)

        # Add the merged masks to the list of circle candidates
        candidates[0].extend(masks)

    # Remove duplicate candidates
    if candidates[0]:
        unique_masks = remove_duplicates(candidates[0])
        # Convert the list of unique masks back to JSON format
        return unique_masks
    else:
        return None
