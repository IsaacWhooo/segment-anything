from pycocotools import mask
import cv2
import numpy as np

def decode_mask_to_polygon(data):
    rle = {
        'counts': data['segmentation']['counts'],
        'size': data['segmentation']['size']
    }

    # Decode RLE to binary mask
    binary_mask = mask.decode(rle)

    # Convert binary mask to polygon
    contours = cv2.findContours(binary_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    polygons = [c.reshape(-1).tolist() for c in contours]

    return polygons

def display_polygon(polygons):
    # Create an empty black image
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Show the polygon in an image
    for polygon in polygons:
        # Reshape to 1 x N x 2 for cv2.polylines
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        # Draw polygon on the image
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the image
    cv2.imshow('Image with Polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Assuming `data` is your JSON object
data = {"segmentation": {"size": [1024, 1024], "counts": "gdRc0h0Po0>D:J5I7J7I7H7L3L4L3L5L3L8I4L4M3M3M2M4L3N2M4M2N3M2N2N2N2N2M4M2N2N2O1N2N1O3N0O2N2N2O1N2N101N2N2O1O0O2O1N1O2N2N101N2O001N101N100O2O001N101M2O101N101O0O101O001N1000001O000O2O00001N100000001O0O1000000000000O101O00000000000000000000000000000000O1000001O0000000000001O0O1000000O2O00000O101N10000O101O0O10001N10001N10001N2N101N100O2N1O2O001N2O001N101N2N2N2N2N2O0O2N101N3M2N2N101N1O2N2N2N3M3M1O2N1O2N3L4M2N2N2N2N2M3N4L2M4L4L3N3L3M3N3K7J4L4K6K4K6J5K8I8C?C=^O[[l5"}, "area": 39462, "bbox": [610, 564, 224, 223], "predicted_iou": 1.02769136428833, "point_coords": [[720.0, 784.0]], "stability_score": 0.9885824918746948, "crop_box": [0, 0, 1024, 1024]}

polygons = decode_mask_to_polygon(data)

print(polygons)
display_polygon(polygons)
