import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def show_mask(mask):
    plt.imshow(mask, cmap='binary')
    plt.axis('off')
    plt.show()

def show_all_masks(mask_list):

    num_masks = len(mask_list)
    if num_masks:
        num_cols = min(4, num_masks)
        num_rows = num_masks // num_cols
        num_rows += num_masks % num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        if num_rows == 1:
            axes = np.expand_dims(axes, 0)

        if num_cols == 1:
            axes = np.expand_dims(axes, -1)

        for idx, mask in enumerate(mask_list):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].imshow(mask, cmap='gray')
            axes[row, col].axis('off')

        for idx in range(num_masks, num_rows*num_cols):
            row = idx // num_cols
            col = idx % num_cols
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        plt.show()

    else:
        print("No mask to display.")


def display_masks_on_image(image, masks):
    # Create a color map for the masks
    colors = plt.cm.hsv(np.linspace(0, 1, len(masks)+1))

    # Display the image
    plt.imshow(image, cmap='gray')

    # Overlay each mask
    for i, mask in enumerate(masks):
        plt.imshow(np.ma.masked_where(mask == 0, mask), cmap='hsv', alpha=0.5)

    plt.axis('off')
    plt.show()

def create_masked_image(image, masks):
    # Convert the original image to RGB
    image = image.convert('RGB')

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Create a new image array for the masks
    mask_image_array = np.zeros(image_array.shape, dtype=np.uint8)

    # For each mask
    for mask in masks:
        # Create a border around the mask
        kernel = np.ones((5,5),np.uint8)
        border = cv2.dilate(mask, kernel, iterations = 1) - mask

        # Add the mask to the mask image array
        mask_image_array[mask > 0] = [255, 0, 0]
        mask_image_array[border > 0] = [255, 255, 255]  # Set the border color to white

    # Create a PIL image from the mask image array
    mask_image = Image.fromarray(mask_image_array)

    # Composite the original image and the mask image
    masked_image = Image.blend(image, mask_image, alpha=0.5)

    return masked_image

