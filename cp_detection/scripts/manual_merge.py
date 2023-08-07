import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the original image
image_folder = '../data'
image_name = 'test_multiple'
img = cv2.imread(os.path.join(image_folder, image_name) + '.jpg')

# Folder where the mask images are stored
mask_folder = './' + image_name + '_filtered'

# Get the list of mask image files
mask_files = os.listdir(mask_folder)

# Prepare a color map with random colors for each mask
num_masks = len(mask_files)
colors = plt.cm.get_cmap('nipy_spectral', num_masks)

# Create a copy of the image for displaying mask indices
img_copy = img.copy()

# Overlay all masks on the original image and display
for i, mask_file in enumerate(mask_files):
    mask_img = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
    color = mcolors.to_rgb(colors(i)[:3])  # Get RGB color
    color = [int(c * 255) for c in color]  # Scale from [0,1] to [0,255]
    img_copy[np.where(mask_img != 0)] = color  # Use a different color for each mask

    # Show the index in the center of the mask
    y, x = np.where(mask_img != 0)
    if len(x) > 0 and len(y) > 0:
        centroid = (sum(x) // len(x), sum(y) // len(y))
        cv2.putText(img_copy, str(i), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

while True:
    # Initialize an empty list to hold masks to be merged
    masks_to_merge = []

    while True:
        mask_index = input("Enter the index of the next mask to merge (or 'm' to merge, 'q' to quit): ")

        # Break the loop if the user wants to merge or quit
        if mask_index.lower() in ['m', 'q']:
            break

        # Load the selected mask
        mask = cv2.imread(os.path.join(mask_folder, mask_files[int(mask_index)]), cv2.IMREAD_GRAYSCALE)
        masks_to_merge.append((int(mask_index), mask))

    if mask_index.lower() == 'm':
        # Merge the masks
        merged_mask = np.bitwise_or.reduce(np.array([mask for index, mask in masks_to_merge]))

        # Save the merged mask
        merged_mask_filename = 'merged_masks_' + '_'.join([str(index) for index, mask in masks_to_merge]) + '.png'
        cv2.imwrite(os.path.join(mask_folder, merged_mask_filename), merged_mask)
        print(f"Merged masks saved to '{merged_mask_filename}'")

        # Delete the individual masks that have been merged
        for index, mask in masks_to_merge:
            os.remove(os.path.join(mask_folder, mask_files[index]))
        print("Individual masks that have been merged were deleted.")

    # Quit if the user entered 'q'
    if mask_index.lower() == 'q':
        print("Exiting...")
        break
