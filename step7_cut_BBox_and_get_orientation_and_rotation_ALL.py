import cv2
import os
import numpy as np
import pickle
import multiprocessing
from ultralytics import YOLO

model = YOLO('yolov8n_wood_segmentation_run8_last.pt')  # load a custom model

width = 3840
height = 2160

def rectify_coordinates(U, V, H):
    '''Get projection of image pixels in real-world coordinates
       given image coordinate matrices and  homography

    Parameters
    ----------
    U : np.ndarray
        NxM matrix containing u-coordinates
    V : np.ndarray
        NxM matrix containing v-coordinates
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing real-world x-coordinates
    np.ndarray
        NxM matrix containing real-world y-coordinates
    '''

    UV = np.vstack((U.flatten(),
                    V.flatten())).T

    # transform image using homography
    XY = cv2.perspectiveTransform(np.asarray([UV]).astype(np.float32), H)[0]

    # reshape pixel coordinates back to image size
    X = XY[:,0].reshape(U.shape[:2])
    Y = XY[:,1].reshape(V.shape[:2])

    return X, Y


def pixels_to_coordinates(homography_file, uv_pixels):
    # Load homography matrix from pickled file
    with open(homography_file, 'rb') as f:
        homography_matrix = pickle.load(f)

    U = np.array([pixel[0] for pixel in uv_pixels])
    V = np.array([pixel[1] for pixel in uv_pixels])

    # Call rectify_coordinates to get real-world coordinates
    X, Y = rectify_coordinates(U, V, homography_matrix)

    transformed_coordinates = np.column_stack((X.flatten(), Y.flatten()))
    
    return transformed_coordinates


def pixels_to_coordinates_old(homography_file, uv_pixels):
    # Load homography matrix from pickled file
    with open(homography_file, 'rb') as f:
        homography_matrix = pickle.load(f)

    transformed_coordinates = []
    for pixel in uv_pixels:
        # Add homogeneous coordinate
        pixel_homogeneous = np.array([pixel[0], pixel[1], 1])

        # Apply homography transformation
        transformed_coordinate = np.dot(homography_matrix, pixel_homogeneous)

        # Normalize the result to get actual coordinates
        transformed_coordinate /= transformed_coordinate[2]

        transformed_coordinates.append(transformed_coordinate[:2])

    return transformed_coordinates

def count_black_pixels(classified_img):
    # Count black pixels
    black_pixel_count = np.sum(classified_img == 0)
    return black_pixel_count

def classify_pixels(img):
    # Read the image
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the average intensity
    average_intensity = np.mean(img)
    
    # Classify pixels as black or white based on the average intensity
    _, classified_img = cv2.threshold(img, average_intensity, 255, cv2.THRESH_BINARY)
    
    # Save the classified image
    #cv2.imwrite(image_bin, classified_img)
    return classified_img

# Function to determine orientation based on edge densities
def determine_orientation(gray,gray_non_rescaled):
    cl_image = classify_pixels(gray_non_rescaled)
    # Define Sobel kernels for diagonal edge detection
    sobel_diag1 = np.array([[ 0,  1, 2],
                            [-1,  0, 1],
                            [-2, -1, 0]])
    sobel_diag12 = np.array([[ 0, -1, -2],
                            [  1,  0, -1],
                            [  2,  1,  0]])

    sobel_diag2 = np.array([[2,  1,  0],
                            [1,  0, -1],
                            [0, -1, -2]])
    sobel_diag22 = np.array([[-2, -1,  0],
                            [ -1,  0,  1],
                            [  0,  1,  2]])

    # Apply Sobel edge detection in diagonal directions
    edge_diag1 = cv2.filter2D(gray, -1, sobel_diag1)
    edge_diag12 = cv2.filter2D(gray, -1, sobel_diag12)

    edge_diag2 = cv2.filter2D(gray, -1, sobel_diag2)
    edge_diag22 = cv2.filter2D(gray, -1, sobel_diag22)
    #print(edge_diag1)
    sum1=sum(sum(edge_diag1))
    sum2=sum(sum(edge_diag2))
    n_white_pix_1 = np.sum(edge_diag1 > 100)
    n_white_pix_12 = np.sum(edge_diag12 > 100)
    n_white_pix_2 = np.sum(edge_diag2 > 100)
    n_white_pix_22 = np.sum(edge_diag22 > 100)

    tot_white_pix_1 = n_white_pix_1 + n_white_pix_12
    tot_white_pix_2 = n_white_pix_2 + n_white_pix_22
    #print(tot_white_pix_1)
    #print(tot_white_pix_2)
    #print('')

    # Compute the magnitude of gradients
    #magnitude_diag1 = np.sqrt(np.square(edge_diag1))
    #magnitude_diag2 = np.sqrt(np.square(edge_diag2))

    # Count non-zero pixels along the diagonals
    #count_diag1 = np.count_nonzero(magnitude_diag1)
    #count_diag2 = np.count_nonzero(magnitude_diag2)

    # Determine orientation based on count of edge pixels
    if tot_white_pix_1 > tot_white_pix_2:
        return ("tlbr",edge_diag1,edge_diag2,n_white_pix_1,n_white_pix_2,cl_image)  # Top left to bottom right
    else:
        return ("trbl",edge_diag1,edge_diag2,n_white_pix_1,n_white_pix_2,cl_image)  # Top right to bottom left

def calculate_average(numbers):
    # Check if the list is not empty to avoid division by zero
    if not numbers:
        raise ValueError("The list of numbers cannot be empty.")

    # Calculate the sum of the numbers in the list
    total_sum = sum(numbers)

    # Calculate the number of elements in the list
    num_elements = len(numbers)

    # Calculate the average
    average = total_sum / num_elements

    return average

def calculate_polygon_area(coordinates):
    n = len(coordinates)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices.")

    # Ensure the coordinates array is a list for easier handling
    if isinstance(coordinates, np.ndarray):
        coordinates = coordinates.tolist()
    
    # Close the polygon by appending the first vertex to the end
    coordinates.append(coordinates[0])

    area = 0
    for i in range(n):
        x_i, y_i = coordinates[i]
        x_next, y_next = coordinates[i + 1]
        area += (x_i * y_next - y_i * x_next)
    
    area = abs(area) / 2
    return area

def get_diameter_from_IM_length_pixelsize(image,piece_length,pixel_size):
    
    #print(image.shape)
    #do prediction
    results = model.predict(image,verbose=False)
    #print('hoi')

    diameters = []
    PIXpolygons = []
    #go thrrough the results
    for result in results:
        boxes = result.boxes
        xywh = boxes.xywh
        BOXpixlength = np.sqrt(xywh[0][2].item() ** 2 + xywh[0][3].item() ** 2 )
        orig_shape = boxes.orig_shape
        TOTpixlength = np.sqrt(orig_shape[0] ** 2 + orig_shape[1] ** 2 )
        lengthpercentage = BOXpixlength/TOTpixlength

        masks = result.masks
        PIXpolygon = masks.xy
        #calculate the area of the polygon
        PIXarea = calculate_polygon_area(PIXpolygon[0])
        #calculate the real world area of the polygon
        RWarea = PIXarea * pixel_size
        #calculate the real world length of the piece of wood in the detected bbox
        RWlength = lengthpercentage * piece_length

        diameter = RWarea / RWlength

        diameters.append(diameter)
        PIXpolygons.append(np.array(PIXpolygon))

    average_diameter = calculate_average(diameters)

    return average_diameter,PIXpolygons


dronenumbers = [[1,1],[1,2],[1,3],[1,4],[1,5],[6,1],[6,2],[6,3],[6,4],[6,5],[16,1],[16,2],[16,3],[16,4],[16,5]]
for dronenumber,vidnumber in dronenumbers:
    #def process_data(data):

    #dronenumber = data[0]
    #vidnumber = data[1]

    # Paths to folders containing images and annotations
    images_folder = 'cut_data/DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber)
    annotations_folder = 'cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/interpolated_detected_linked_IDs'
    homographies_folder = 'cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/correct_homs_18jan_all'

    new_annotations_folder = 'cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodvolume'
    new_annotations_folder = '/media/jean-pierre/PortableSSD/work_scratch_7mei/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodlength_wooddiameter_woodvolume'
    new_annotations_folder = 'cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodlength_wooddiameter_woodvolume'

    os.makedirs(new_annotations_folder, exist_ok=True)

    # Output folder for saving bounding boxes
    output_folder = '/media/jean-pierre/PortableSSD/work_scratch_7mei/test_BBOX'
    output_folder_ALL = '/media/jean-pierre/PortableSSD/work_scratch_7mei/test_BBOX_all'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_ALL, exist_ok=True)

    new_folder = 'cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_orientation_rotation'


    # Process each annotation file
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith('.txt'):
            # Load bounding box annotations
            with open(os.path.join(annotations_folder, annotation_file), 'r') as file:
                annotations = file.readlines()

            # Load corresponding image
            image_filename = annotation_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_folder, image_filename)
            #print(image_path)
            image = cv2.imread(image_path)
            hom_file = os.path.join(homographies_folder,annotation_file.replace('.txt','_hom.p'))

            # Process each bounding box in the annotation file
            for idx, annotation in enumerate(annotations):
                parts = annotation.split()
                x_center, y_center, box_width, box_height = map(float, parts[1:5])

                # Determine the larger dimension
                max_dim = max(box_width*3840, box_height*2160)

                # Calculate scaling factors for resizing
                scaling_factor_x = max_dim / box_width*3840
                scaling_factor_y = max_dim / box_height*2160

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize the bounding box region to make it square
                region_of_interest_RGB = image_rgb[int((y_center - (box_height / 2)) * image.shape[0]):
                                           int((y_center + (box_height / 2)) * image.shape[0]),
                                           int((x_center - (box_width / 2)) * image.shape[1]):
                                           int((x_center + (box_width / 2)) * image.shape[1])]
                

                # Resize the bounding box region to make it square
                region_of_interest = image[int((y_center - (box_height / 2)) * image.shape[0]):
                                           int((y_center + (box_height / 2)) * image.shape[0]),
                                           int((x_center - (box_width / 2)) * image.shape[1]):
                                           int((x_center + (box_width / 2)) * image.shape[1])]
                region_of_interest_rescaled = cv2.resize(region_of_interest, (int(max_dim), int(max_dim)))

                # Convert to grayscale
                gray_non_rescaled = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(region_of_interest_rescaled, cv2.COLOR_BGR2GRAY)

                # Determine orientation based on Sobel edge detection
                orientation,edge_diag1,edge_diag2,sum1,sum2,cl_image = determine_orientation(gray,gray_non_rescaled)

                # Save the bounding box with appropriate name
                output_filename = f'{os.path.splitext(image_filename)[0]}_BBox_{idx+1}_{orientation}.jpg'
                output_path = os.path.join(output_folder, output_filename)

                output_path_ALL = os.path.join(output_folder_ALL, output_filename)

                black_pixel_count = count_black_pixels(cl_image)

                

                pix_center = (x_center*width, y_center*height)
                pix_center_plus_one = (x_center*width + 1, y_center*height + 1)

                pix_box_width = box_width * width
                pix_box_height = box_height * height
                pix_topleft = (x_center*width - 0.5*pix_box_width, y_center*height - 0.5*pix_box_height)
                pix_topright = (x_center*width + 0.5*pix_box_width, y_center*height - 0.5*pix_box_height)
                pix_bottomleft = (x_center*width - 0.5*pix_box_width, y_center*height + 0.5*pix_box_height)
                pix_bottomright = (x_center*width + 0.5*pix_box_width, y_center*height + 0.5*pix_box_height)
                #print('box_width', box_width)
                #print(pix_center)
                #print(pix_topleft)
                #print(pix_bottomright)
                transformed_coordinates = pixels_to_coordinates(hom_file,[pix_center,pix_topleft,pix_topright,pix_bottomleft,pix_bottomright,pix_center_plus_one])
                pix_size_x = transformed_coordinates[0][0] - transformed_coordinates[5][0]
                pix_size_y = transformed_coordinates[5][1] - transformed_coordinates[0][1]
                
                if orientation == "tlbr":
                    piece_length = np.sqrt((transformed_coordinates[1][0]-transformed_coordinates[4][0])**2 + (transformed_coordinates[1][1]-transformed_coordinates[4][1])**2 )
                if orientation == "trbl":
                    piece_length = np.sqrt((transformed_coordinates[2][0]-transformed_coordinates[3][0])**2 + (transformed_coordinates[2][1]-transformed_coordinates[3][1])**2 )
                piece_tot_surface = np.absolute(pix_size_x) * np.absolute(pix_size_y) * black_pixel_count
                piece_diameter = piece_tot_surface / piece_length

                pixel_size = np.absolute(pix_size_x) * np.absolute(pix_size_y)
                #print(pixel_size)

                #print(region_of_interest_RGB)
                #results = model.predict(image_rgb)
                #print('hoi')
                try:
                    piece_diameter,PIXpolygons = get_diameter_from_IM_length_pixelsize(region_of_interest_RGB,piece_length,pixel_size)
                except:
                    PIXpolygons=None
                    cv2.imwrite(output_path.replace('.jpg','_not_rescaled_diam.png'), region_of_interest)
                    print('no wood detected by segmentation algorithm')
                #print(pixel_size)
                #if piece_diameter < 0:
                #    print(annotation_file)

                piece_volume = piece_length * np.pi * ( piece_diameter / 2 ) ** 2
                if piece_diameter < 0:
                    print('SMALLER THEN 0 Diameter')
                    print(piece_diameter)
                    print(image_path)
                #print('piece diameter',piece_diameter)
                #print('piece volume',piece_volume)

                #print('transformed_coordinates', transformed_coordinates)
                #print('length: ', piece_length )
                #print('uitersten',pixels_to_coordinates(hom_file,[(0,0),(3840,2160)]))

                #print('black pixels ', black_pixel_count)
                new_file_name = os.path.join(new_annotations_folder,annotation_file)
                #print(new_file_name)
                
                ###################################WRITE#####################################
                with open(new_file_name, 'a') as file:
                    file.write(annotation.split('\n')[0]+' '+str(orientation)+' ['+str(transformed_coordinates[0][0])+str(',')+str(transformed_coordinates[0][1])+'] ['+str(transformed_coordinates[1][0])+str(',')+str(transformed_coordinates[1][1])+'] ['+str(transformed_coordinates[2][0])+str(',')+str(transformed_coordinates[2][1])+'] ['+str(transformed_coordinates[3][0])+str(',')+str(transformed_coordinates[3][1])+'] ['+str(transformed_coordinates[4][0])+str(',')+str(transformed_coordinates[4][1])+'] '+str(piece_length)+' '+str(piece_diameter)+' '+str(piece_volume)+'\n')
                
                #cv2.imwrite(output_path, region_of_interest_rescaled)
                #region_of_interest_RGB
                '''
                if PIXpolygons != None:
                    for PIXpolygon in PIXpolygons:
                        pts = np.round(np.int32(PIXpolygon[0]))
                        #print(pts)
                        #pts = np.array([[0.93, 0.123], [1, 1], 
                        #    [2, 2], [3, 1]],
                        #   np.int32)
                        #print(pts)

                        pts = pts.reshape((-1, 1, 2))
                        #print(pts)
                        isClosed = True
                        thickness = 1
                        color = (255, 0, 0)
                        
                        image_with = cv2.polylines(region_of_interest_RGB, [pts], isClosed, color, thickness)
                    
                    cv2.imwrite(output_path_ALL.replace('.jpg','_not_rescaled_diam.png'), image_with)
                '''
                #cv2.imwrite(output_path.replace('.jpg','_not_rescaled_diam.png'), region_of_interest)

                #cv2.imwrite(output_path.replace('.jpg','_edge1_'+str(sum1)+'_'+str(sum2)+'.jpg'), edge_diag1)
                #cv2.imwrite(output_path.replace('.jpg','_edge2_'+str(sum1)+'_'+str(sum2)+'.jpg'), edge_diag2)
                #cv2.imwrite(output_path.replace('.jpg','_not_rescaled_diam_simpleclassified.png'), cl_image)

        #print(annotation_file)

#dronenumbers = [1,6,16]

dronenumbers = [[1,1],[1,2],[1,3],[1,4],[1,5],[6,1],[6,2],[6,3],[6,4],[6,5],[16,1],[16,2],[16,3],[16,4],[16,5]]

if __name__ == "__main__":
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=1) as pool:
        # Map the processing function to the list of SVG files
        #print(stagered_drone_ims)
        pool.map(process_data, dronenumbers)