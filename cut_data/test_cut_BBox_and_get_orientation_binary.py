import cv2
import os
import numpy as np

# Paths to folders containing images and annotations
images_folder = 'DCIM-drone1_drone1vid1'
annotations_folder = 'DCIM-drone1/drone1vid1/interpolated_detected_linked_IDs'

# Output folder for saving bounding boxes
output_folder = '/media/jean-pierre/PortableSSD/work/test_BBOX'
os.makedirs(output_folder, exist_ok=True)


import cv2
import os
import numpy as np

def classify_pixels(img):
    # Read the image
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the average intensity
    average_intensity = 1 * np.mean(img)
    
    # Classify pixels as black or white based on the average intensity
    _, classified_img = cv2.threshold(img, average_intensity, 255, cv2.THRESH_BINARY)
    
    # Save the classified image
    #cv2.imwrite(image_bin, classified_img)
    return classified_img

# Function to determine orientation based on edge densities
def determine_orientation(gray):
    cl_iamge = classify_pixels(gray)
    # Define Sobel kernels for diagonal edge detection
    sobel_diag1 = np.array([[0, 1, 2],
                             [-1, 0, 1],
                             [-2, -1, 0]])
    sobel_diag2 = np.array([[2, 1, 0],
                             [1, 0, -1],
                             [0, -1, -2]])

    # Apply Sobel edge detection in diagonal directions
    edge_diag1 = cv2.filter2D(gray, -1, sobel_diag1)
    edge_diag2 = cv2.filter2D(gray, -1, sobel_diag2)
    #print(edge_diag1)
    sum1=sum(sum(edge_diag1))
    sum2=sum(sum(edge_diag2))
    n_white_pix_1 = np.sum(edge_diag1 > 100)
    n_white_pix_2 = np.sum(edge_diag2 > 100)

    # Compute the magnitude of gradients
    magnitude_diag1 = np.sqrt(np.square(edge_diag1))
    magnitude_diag2 = np.sqrt(np.square(edge_diag2))

    # Count non-zero pixels along the diagonals
    count_diag1 = np.count_nonzero(magnitude_diag1)
    count_diag2 = np.count_nonzero(magnitude_diag2)

    # Determine orientation based on count of edge pixels
    if n_white_pix_1 > n_white_pix_2:
        return ("tlbr",edge_diag1,edge_diag2,n_white_pix_1,n_white_pix_2,cl_iamge)  # Top left to bottom right
    else:
        return ("trbl",edge_diag1,edge_diag2,n_white_pix_1,n_white_pix_2,cl_iamge)  # Top right to bottom left


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

            # Determine orientation based on Sobel edge detection
            orientation,edge_diag1,edge_diag2,sum1,sum2,cl_iamge = determine_orientation(gray)

            # Save the bounding box with appropriate name
            output_filename = f'{os.path.splitext(image_filename)[0]}_BBox_{idx+1}_{orientation}.jpg'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, region_of_interest_rescaled)
            cv2.imwrite(output_path.replace('.jpg','_edge1_'+str(sum1)+'_'+str(sum2)+'.jpg'), edge_diag1)
            cv2.imwrite(output_path.replace('.jpg','_edge2_'+str(sum1)+'_'+str(sum2)+'.jpg'), edge_diag2)
            cv2.imwrite(output_path.replace('.jpg','_classified_'+str(sum1)+'_'+str(sum2)+'.jpg'), cl_iamge)