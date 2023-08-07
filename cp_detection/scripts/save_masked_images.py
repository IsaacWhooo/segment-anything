import os
import json
from PIL import Image
from tqdm import tqdm
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.json_utils import json2list
from utils.display_utils import create_masked_image
import glob

# Get all filtered json files
filtered_json_folder = "../data/filtered_json"
filtered_json_files = glob.glob(os.path.join(filtered_json_folder, "*.json"))

# Specify the folder to save the images
output_folder = "../data/masked_images"

# Make sure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Loop over each json file
for json_file_path in tqdm(filtered_json_files, desc='Processing JSON files'):
    # Extract the image name from the json file name
    image_name = os.path.basename(json_file_path)[9:-5]

    # Specify the path to the original image
    image_file_path = "../original_images/" + image_name + ".jpg"

    # Load the JSON data
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Convert the JSON data to a list of numpy arrays (masks)
    annotations = json2list(json_data)

    # Load the original image
    original_image = Image.open(image_file_path)

    # Display the masks on the original image
    masked_image = create_masked_image(original_image, annotations)

    # Save the masked image to the output folder
    output_path = os.path.join(output_folder, f"masked_{image_name}.jpg")
    masked_image.save(output_path)
