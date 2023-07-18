from pycocotools import mask
import numpy as np
import json
import os
import base64
def json2list(json_data):
    ndarray_list = []
    for item in json_data:
        segmentation = item['segmentation']
        rle = {
            'counts': segmentation['counts'],
            'size': segmentation['size']
        }
        ndarray_list.append(mask.decode(rle))
    return ndarray_list


def list2json(mask_list):
    """
    Convert a list of binary masks into JSON format.

    Args:
    mask_list : list
        List of binary masks

    Returns:
    json_masks : str
        JSON formatted string of binary masks
    """

    # Initialize an empty dictionary to store the masks
    mask_dict = {}

    # Loop through each mask in the list
    for i, binary_mask in enumerate(mask_list):
        # Convert the binary mask to RLE and store it in the dictionary
        rle = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

        # Convert the 'counts' field to a string
        rle['counts'] = base64.b64encode(rle['counts']).decode()

        mask_dict[str(i)] = rle

    # Convert the dictionary to a JSON formatted string
    json_masks = json.dumps(mask_dict)

    return json_masks

def save_filtered_json(json_data, original_file_path, output_folder):
    """
    Save the filtered JSON data to a new file.

    Args:
    json_data: str
        The filtered JSON data.
    original_file_path: str
        The file path of the original JSON file.
    output_folder: str
        The path of the folder where the filtered JSON file will be saved.
    """
    # Get the file name of the original JSON file
    original_file_name = os.path.basename(original_file_path)

    # Create the file name for the filtered JSON file
    filtered_file_name = "filtered_" + original_file_name

    # Create the file path for the filtered JSON file
    filtered_file_path = os.path.join(output_folder, filtered_file_name)

    # Save the filtered JSON data to the new file
    with open(filtered_file_path, 'w') as json_file:
        json_file.write(json_data)



