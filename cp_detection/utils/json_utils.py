from pycocotools import mask
import numpy as np
import json
import os
import base64
from label_studio_converter import brush

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

    # Initialize an empty list to store the masks
    mask_list_json = []

    # Loop through each mask in the list
    for i, binary_mask in enumerate(mask_list):
        # Convert the binary mask to RLE and store it in the dictionary
        rle = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

        # Convert the 'counts' field to a string
        rle['counts'] = rle['counts'].decode('utf-8')

        # Store the size and counts in a dictionary
        segmentation_dict = {
            'size': list(rle['size']),
            'counts': rle['counts']
        }

        # Append the segmentation dictionary to the mask list
        mask_list_json.append({'segmentation': segmentation_dict})

    # Convert the list of dictionaries to a JSON formatted string
    json_masks = json.dumps(mask_list_json)

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


def masks2rle(masks):
    """
    Convert a list of masks into a list of RLE representations.

    Args:
    masks: list of np.array
        A list of masks.

    Returns:
    rles: list of dict
        A list of RLE representations of the masks.
    """
    rles = []
    for mask in masks:
        rle = brush.mask2rle(mask)
        rles.append(rle)
    return rles

def save_label_studio_task(rles, image_path, output_folder):
    """
    Save a task that can be imported into Label Studio.

    Args:
    rles: list of dict
        A list of RLE representations of the masks.
    image_path: str
        The path to the image file.
    output_folder: str
        The folder where the task will be saved.
    """
    for i, rle in enumerate(rles):
        task = {
            'data': {'image': image_path},
            'predictions': [
                {
                    'result': [
                        {
                            'from_name': 'tag',
                            'to_name': 'image',
                            'type': 'brushlabels',
                            'value': {
                                'brushlabels': ['cp'],
                                'image_rotation': 0,
                                'rle': rle
                            }
                        }
                    ],
                    'score': None,
                    'model_version': None
                }
            ]
        }
        task_file_name = os.path.join(output_folder, f'ls_{os.path.splitext(os.path.basename(image_path))[0]}_{i}.json')
        with open(task_file_name, 'w') as f:
            json.dump(task, f)