from PIL import Image
import os
import sys
import json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.json_utils import json2list
from utils.display_utils import display_masks_on_image

filtered_json_name = "filtered_L15-1180E-0931N_tile_3_2.json"
image_name = filtered_json_name[9:-5]

image_file_path = "../original_images/" + image_name + ".jpg"
json_file_path = "../filtered_json/" + filtered_json_name
original_json_file_path = "../data/input_json/" + image_name + ".json"

with open(json_file_path, 'r') as json_file:
    json_data1 = json.load(json_file)

annotations = json2list(json_data1)
print(f"Number of masks left: {len(annotations)}.")

with open(original_json_file_path, 'r') as json_file2:
    json_data2 = json.load(json_file2)

full_annotations = json2list(json_data2)
print(f"Number of masks generated by SAM: {len(full_annotations)}.")

original_image = Image.open(image_file_path)

display_masks_on_image(original_image,annotations)

display_masks_on_image(original_image,full_annotations)

