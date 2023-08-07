import os
import json
import csv
import sys

from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.json_utils import json2list
from utils.display_utils import display_masks_on_image

# Folder containing images and JSON files
image_folder = "../original_images"
json_folder = "../data/filtered_json"

# Output CSV file
output_file = "validation_results.csv"

# Open the output CSV file
with open(output_file, "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image_name", "missing_masks", "wrong_masks", "total_masks"])

    # Get all JSON files
    json_files = [file for file in os.listdir(json_folder) if file.endswith(".json")]
    num_files = len(json_files)

    # For each JSON file
    for i, json_file in enumerate(json_files, start=1):
        print(f"Processing file {i} of {num_files}: {json_file}")

        while True:  # Loop until user confirms
            # Load the image and JSON file
            image_name = json_file[9:-5]
            image = Image.open(os.path.join(image_folder, image_name + ".jpg"))
            with open(os.path.join(json_folder, json_file), "r") as f:
                masks = json2list(json.load(f))
                mask_num = len(masks)

            # Display the image with masks
            display_masks_on_image(image, masks)

            # Prompt the user to input the number of missing and wrong masks
            missing = input("Enter number of missing masks: ")
            wrong = input("Enter number of wrong masks: ")

            # Prompt the user to confirm
            print(f"Missing masks: {missing}, Wrong masks: {wrong}")
            confirm = input("Confirm? (y/n): ")

            if confirm.lower() == "y":
                # Write the result to the CSV file
                writer.writerow([image_name, missing, wrong, mask_num])
                break  # Exit the loop

        print(f"Finished processing file {i} of {num_files}: {json_file}")
        print("-" * 50)