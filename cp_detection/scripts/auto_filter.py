import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.image_utils import watershed, remove_small_objects, detect_circles_and_semi_circles, merge_and_segment, remove_duplicates
from utils.json_utils import json2list, list2json, save_filtered_json
import glob
import json

# Specify the path of the input folder
input_folder = "../input_json"

# Specify the path of the output folder
output_folder = "../filtered_json"

# Get a list of all JSON files in the input folder
json_files = glob.glob(os.path.join(input_folder, "*.json"))

print(f"Find {len(json_files)} files.")

def auto_filter(json_input: str, min_circular_ratio: float = 0.92, max_circular_ratio: float = 1.05,
                min_semi_ratio: float = 0.45, max_semi_ratio: float = 0.55, min_size_ratio: float = 0.01) -> str:
    """
    Apply an automated filter to the annotations in a JSON string.

    Args:
    json_input: str
        A JSON string of annotations.
    min_circular_ratio: float
        The minimum area ratio for a region to be considered a circle.
    max_circular_ratio: float
        The maximum area ratio for a region to be considered a circle.
    min_semi_ratio: float
        The minimum area ratio for a region to be considered a semi-circle.
    max_semi_ratio: float
        The maximum area ratio for a region to be considered a semi-circle.
    min_size_ratio: float
        The minimum object size as a ratio of the total image size.

    Returns:
    json_output: str
        A JSON string of filtered annotations.
    """
    # Convert the JSON string into a list of numpy arrays
    arrays = json2list(json.loads(json_input))

    # Initialize a list to store the candidates
    candidates = [[], []]

    # For each array
    for array in arrays:
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

    # Merge and segment the semi-circle candidates
    masks = merge_and_segment(candidates[1], min_circular_ratio, max_circular_ratio)

    # Add the merged masks to the list of circle candidates
    candidates[0].extend(masks)

    # Remove duplicate candidates
    unique_masks = remove_duplicates(candidates[0])

    # Convert the list of unique masks back to JSON format
    json_output = list2json(unique_masks)

    return json_output


# Process each JSON file
for i, json_file_path in enumerate(json_files):
    print(f"Processing file: {json_file_path}, {i+1} of {len(json_files)}.")  # Print the path of the current file

    # Open the JSON file and read the data
    with open(json_file_path, 'r') as json_file:
        json_data = json_file.read()

    # Apply the automated filter to the JSON data
    print("Applying filter...")  # Print a message before the filter is applied
    filtered_data = auto_filter(json_data)
    print("Filter applied.")  # Print a message after the filter is applied

    # Save the filtered JSON data to a new file in the output folder
    print("Saving filtered data...")  # Print a message before the data is saved
    save_filtered_json(filtered_data, json_file_path, output_folder)
    print("Filtered data saved.")  # Print a message after the data is saved

print("All files processed.")  # Print a message after all files have been processed
