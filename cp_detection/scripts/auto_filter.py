import os
import sys
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.image_utils import watershed, remove_small_objects, detect_circles_and_semi_circles, merge_and_segment, remove_duplicates
from utils.json_utils import json2list, list2json, save_filtered_json
import glob
import json
from utils.auto_filter_utils import auto_filter

# Specify the path of the input folder
input_folder = "../data/input_json"

# Specify the path of the output folder
output_folder = "../data/filtered_json"

# Get a list of all JSON files in the input folder
json_files = glob.glob(os.path.join(input_folder, "*.json"))

total_file_number = len(json_files)
print(f"Find {total_file_number} files.")

counter = 0

# Process each JSON file
for i, json_file_path in tqdm(enumerate(json_files), total=len(json_files), file=sys.stdout):
    # print(f"Processing file: {json_file_path}, {i+1} of {len(json_files)}.")  # Print the path of the current file

    # Open the JSON file and read the data
    with open(json_file_path, 'r') as json_file:
        json_data = json_file.read()

    # Apply the automated filter to the JSON data
    # print("Applying filter...")  # Print a message before the filter is applied
    filtered_data = auto_filter(json_data)
    # print("Filter applied.")  # Print a message after the filter is applied
    if filtered_data:
        filtered_json = list2json(filtered_data)
        save_filtered_json(filtered_json, json_file_path, output_folder)
        print("Masks saved.")
        return 1
    else:
        print("No mask found in this file.")
        return 0

print(f"Processed {total_file_number}. Saved {counter} files.")  # Print a message after all files have been processed

