from skimage.measure import regionprops
import cv2
import numpy as np
from skimage import morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from typing import List, Tuple
import warnings

def watershed(mask_image: np.ndarray) -> np.ndarray:
    """
    Performs a watershed segmentation on the image and returns the segmented labels

    Args:
    mask_image: ndarray
        The mask image to segment

    Returns:
    labels: np.ndarray
        The segmented labels
    """
    # Threshold the image to isolate the circles
    thresholded = mask_image > np.mean(mask_image)

    # Compute the distance transform
    distance = ndi.distance_transform_edt(thresholded)

    # Find the peak local maxima
    coordinates = peak_local_max(distance, labels=thresholded, min_distance=20)

    # Create an empty image to store the markers
    markers = np.zeros(distance.shape, dtype=np.uint8)

    # Draw the markers
    for i, (r, c) in enumerate(coordinates):
        markers[r, c] = i + 1  # Add 1 to avoid confusion with the background

    # Perform the watershed segmentation
    labels = segmentation.watershed(-distance, markers, mask=thresholded)

    return labels

def remove_small_objects(labels: np.ndarray, min_size_ratio: float) -> np.ndarray:
    """
    Removes small objects from the segmented labels based on the minimum size ratio.

    Args:
    labels: ndarray
        The segmented labels from which to remove small objects
    min_size_ratio: float
        The minimum object size as a ratio of the total image size

    Returns:
    labels: ndarray
        The segmented labels with small objects removed
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # Define the minimum object size (as a fraction of the total image size)
    min_object_size = min_size_ratio * labels.size

    # Remove small objects
    labels = morphology.remove_small_objects(labels, min_size=min_object_size)

    return labels

def detect_circles(labels: np.ndarray, min_circular_ratio: float, max_circular_ratio: float) -> List[int]:
    """
    Detects circular regions in the labels.

    Args:
    labels: ndarray
        The segmented labels in which to detect circles
    min_circular_ratio: float
        The minimum area ratio for a region to be considered a circle
    max_circular_ratio: float
        The maximum area ratio for a region to be considered a circle

    Returns:
    circle_labels: list of int
        The list of labels of detected circles
    """
    # Calculate the properties of the regions
    regions = regionprops(labels)

    # Initialize a list to store the labels of circles
    circle_labels = []

    # For each region
    for region in regions:
        if region.area > 0:  # Ignore background
            # Get the coordinates of all pixels in the region
            coords = np.column_stack(region.coords)

            # Transpose the coordinates
            coords_t = np.transpose(coords)

            # Calculate the minimum enclosing circle using OpenCV
            (x, y), radius = cv2.minEnclosingCircle(coords_t.astype(np.float32))

            # Calculate the area of the minimum enclosing circle
            circle_area = np.pi * radius ** 2

            # Compare the two areas
            ratio = region.area / circle_area

            # If the ratio is within the circularity thresholds, add the region label to the list
            if min_circular_ratio <= ratio <= max_circular_ratio:
                circle_labels.append(region.label)

    return circle_labels


def detect_circles_and_semi_circles(labels: np.ndarray, min_circular_ratio: float,
                                    max_circular_ratio: float, min_semi_ratio: float,
                                    max_semi_ratio: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Detects circle and semi-circle regions in the labels.

    Args:
    labels: ndarray
        The segmented labels in which to detect circles and semi-circles
    min_circular_ratio: float
        The minimum area ratio for a region to be considered a circle
    max_circular_ratio: float
        The maximum area ratio for a region to be considered a circle
    min_semi_ratio: float
        The minimum area ratio for a region to be considered a semi-circle
    max_semi_ratio: float
        The maximum area ratio for a region to be considered a semi-circle

    Returns:
    circle_masks: list of ndarray
        The list of binary masks of detected circles
    semi_circle_masks: list of ndarray
        The list of binary masks of detected semi-circles
    """
    # Calculate the properties of the regions
    regions = regionprops(labels)

    # Initialize lists to store the masks of circles and semi-circles
    circle_masks = []
    semi_circle_masks = []

    # For each region
    for region in regions:
        if region.area > 0:  # Ignore background
            # Get the coordinates of all pixels in the region
            coords = np.column_stack(region.coords)

            # Transpose the coordinates
            coords_t = np.transpose(coords)

            # Calculate the minimum enclosing circle using OpenCV
            (x, y), radius = cv2.minEnclosingCircle(coords_t.astype(np.float32))

            # Calculate the area of the minimum enclosing circle
            circle_area = np.pi * radius ** 2

            # Compare the two areas
            ratio = region.area / circle_area

            # Create a new binary mask for the region
            mask = np.zeros_like(labels)
            mask[coords[0, :], coords[1, :]] = 1 # Transpose the coordinates here

            # If the ratio is within the circularity thresholds, add the mask to the corresponding list
            if min_circular_ratio <= ratio <= max_circular_ratio:
                circle_masks.append(mask)
            elif min_semi_ratio <= ratio <= max_semi_ratio:
                semi_circle_masks.append(mask)

    return circle_masks, semi_circle_masks




def merge_and_segment(semi_circles: List[np.ndarray], min_circular_ratio: float, max_circular_ratio: float) -> List[np.ndarray]:
    """
    Merge all semi-circle images into a single image and then segment it into individual masks. Detect and keep only circles.

    Args:
    semi_circles: list of ndarray
        List of semi-circle images as 2D numpy arrays
    min_circular_ratio: float
        The minimum area ratio for a region to be considered a circle
    max_circular_ratio: float
        The maximum area ratio for a region to be considered a circle

    Returns:
    masks: list of ndarray
        The list of segmented masks
    """
    # Merge all images
    merged_image = np.max(semi_circles, axis=0)

    # Perform morphological closing
    closed_image = morphology.closing(merged_image, morphology.disk(5))

    # Perform watershed segmentation
    segmented_image = watershed(closed_image)

    # Detect circles in the segmented image
    circle_regions = detect_circles(segmented_image, min_circular_ratio, max_circular_ratio)

    # Create a mask for each circle region
    masks = [np.isin(segmented_image, region).astype(int) for region in circle_regions]

    return masks

def remove_duplicates(masks: List[np.ndarray], overlap_threshold: float = 0.8) -> List[np.ndarray]:
    """
    Removes duplicate masks based on the overlap threshold.

    Args:
    masks: list of ndarray
        List of binary masks as 2D numpy arrays
    overlap_threshold: float
        The overlap threshold for two masks to be considered duplicates

    Returns:
    unique_masks: list of ndarray
        The list of unique masks
    """
    # Initialize the list of unique masks with the first mask
    unique_masks = [masks[0]]

    # For each mask
    for mask in masks[1:]:
        # Initialize a flag to indicate whether the mask is a duplicate
        is_duplicate = False

        # Compare the mask to each unique mask
        for i, unique_mask in enumerate(unique_masks):
            # Calculate the overlap as the intersection over union (IoU)
            intersection = np.logical_and(mask, unique_mask).sum()
            union = np.logical_or(mask, unique_mask).sum()
            overlap = intersection / union

            # If the overlap exceeds the threshold, mark the mask as a duplicate
            if overlap > overlap_threshold:
                is_duplicate = True
                # If the current mask is larger, replace the unique_mask with the current mask
                if mask.sum() > unique_mask.sum():
                    unique_masks[i] = mask
                break

        # If the mask is not a duplicate, add it to the list of unique masks
        if not is_duplicate:
            unique_masks.append(mask)

    return unique_masks




