import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.json_utils import json2list, list2json, save_filtered_json
from utils.auto_filter_utils import auto_filter
import glob


def process_file(json_file_path, output_folder):
    # Open the JSON file and read the data
    with open(json_file_path, 'r') as json_file:
        json_data = json_file.read()

    # Apply the automated filter to the JSON data
    filtered_data = auto_filter(json_data)

    if filtered_data:
        filtered_json = list2json(filtered_data)
        save_filtered_json(filtered_json, json_file_path, output_folder)
        print("Masks saved.")
        return 1
    else:
        print("No mask found in this file.")
        return 0

if __name__ == '__main__':
    # Specify the path of the input folder
    input_folder = "../data/input_json_0728"

    # Specify the path of the output folder
    output_folder = "../data/filtered_json_0728"

    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, "*.json"))

    total_file_number = len(json_files)
    print(f"Find {total_file_number} files.")

    # Use a ProcessPoolExecutor to process the files in parallel
    with ProcessPoolExecutor() as executor:
        process_func = partial(process_file, output_folder=output_folder)
        results = list(tqdm(executor.map(process_func, json_files), total=len(json_files)))

    counter = sum(results)
    print(f"Processed {total_file_number}. Saved {counter} files.")
