import cv2
import os
import numpy as np
from skimage.segmentation import active_contour
from sklearn.decomposition import PCA

# Paths to folders containing images and annotations
images_folder = 'DCIM-drone1_drone1vid1'
annotations_folder = 'DCIM-drone1/drone1vid1/interpolated_detected_linked_IDs'

# Output folder for saving bounding boxes
output_folder = '/media/jean-pierre/PortableSSD/work/test_BBOX'
os.makedirs(output_folder, exist_ok=True)


# Function to determine orientation using PCA
def determine_orientation_PCA(image):
    # Define initial contour as a circular contour
    init_radius = min(image.shape[0], image.shape[1]) / 2 - 2
    init_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    angles = np.linspace(0, 2*np.pi, 100)
    init = init_center + init_radius * np.column_stack((np.cos(angles), np.sin(angles)))

    # Apply Chain-Vese segmentation
    snake = active_contour(image, init, alpha=0.1, beta=1.0, gamma=0.01)

    # Perform PCA on segmented points
    pca = PCA(n_components=2)
    pca.fit(image)
    angle_rad = np.arctan2(*pca.components_[0])  # Angle in radians
    angle_deg = np.degrees(angle_rad)  # Angle in degrees

    # Determine orientation based on angle
    if angle_deg >= -45 and angle_deg < 45:
        return ("tlbr",snake)  # Top left to bottom right
    else:
        return ("trbl",snake)  # Top right to bottom left


# Process each annotation file
for annotation_file in os.listdir(annotations_folder):
    if annotation_file.endswith('.txt'):
        # Load bounding box annotations
        with open(os.path.join(annotations_folder, annotation_file), 'r') as file:
            annotations = file.readlines()

        # Load corresponding image
        image_filename = annotation_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_folder, image_filename)
        image = cv2.imread(image_path)

        # Process each bounding box in the annotation file
        for idx, annotation in enumerate(annotations):
            parts = annotation.split()
            x_center, y_center, box_width, box_height = map(float, parts[1:5])

            # Determine the larger dimension
            max_dim = max(box_width*3920, box_height*2160)

            # Calculate scaling factors for resizing
            scaling_factor_x = max_dim / box_width
            scaling_factor_y = max_dim / box_height

            # Resize the bounding box region to make it square
            region_of_interest = image[int((y_center - (box_height / 2)) * image.shape[0]):
                                       int((y_center + (box_height / 2)) * image.shape[0]),
                                       int((x_center - (box_width / 2)) * image.shape[1]):
                                       int((x_center + (box_width / 2)) * image.shape[1])]
            region_of_interest_rescaled = cv2.resize(region_of_interest, (int(max_dim), int(max_dim)))

            # Convert to grayscale
            gray = cv2.cvtColor(region_of_interest_rescaled, cv2.COLOR_BGR2GRAY)

            # Determine orientation using PCA
            orientation,snake = determine_orientation_PCA(gray)

            # Save the bounding box with appropriate name
            output_filename = f'{os.path.splitext(image_filename)[0]}_BBox_{idx+1}_{orientation}.jpg'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, region_of_interest_rescaled)

