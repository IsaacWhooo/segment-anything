import os
import sys
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.image_utils import watershed, remove_small_objects, detect_circles_and_semi_circles, merge_and_segment, remove_duplicates
from utils.json_utils import json2list, masks2rle, save_label_studio_task
import glob
import json

# Specify the path of the input folder
input_folder = "../test_data/test_input_json"

# Specify the path of the output folder
output_folder = "../test_data/test_output_json"

# Get a list of all JSON files in the input folder
json_files = glob.glob(os.path.join(input_folder, "*.json"))

total_file_number = len(json_files)
print(f"Find {total_file_number} files.")

def auto_filter(json_input: str, min_circular_ratio: float = 0.9, max_circular_ratio: float = 1.05,
                min_semi_ratio: float = 0.45, max_semi_ratio: float = 0.55, min_size_ratio: float = 0.005,
                second_min_circular_ratio: float = 0.7) -> str:

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
        rles = masks2rle(unique_masks)
        return rles
    else:
        return None


counter = 0
for i, json_file_path in tqdm(enumerate(json_files), total=len(json_files), file=sys.stdout):
    # Open the JSON file and read the data
    with open(json_file_path, 'r') as json_file:
        json_data = json_file.read()

    # Apply the automated filter to the JSON data
    rles = auto_filter(json_data)
    if rles:
        image_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.jpg'
        image_url = f"https://raw.githubusercontent.com/IsaacWhooo/segment-anything/main/cp_detection/test_data/{image_file_name}"

        save_label_studio_task(rles, image_url, output_folder)
        print("Masks saved.")
        counter += 1

    else:
        print("No mask found in this file.")

print(f"Processed {total_file_number}. Saved {counter} files.")  # Print a message after all files have been processed

