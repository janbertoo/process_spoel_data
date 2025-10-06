import os
import cv2
from multiprocessing import Pool

# Function to read detections from text file
def read_detections(file_path):
    with open(file_path, 'r') as file:
        detections = []
        for line in file:
            parts = line.strip().split()
            class_name = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5])
            id_number = int(parts[6])
            detections.append((class_name, x_center, y_center, width, height, confidence, id_number))
        return detections

# Function to draw bounding boxes on image
def draw_boxes(image_path, detections, output_folder):
    image = cv2.imread(image_path)
    for detection in detections:
        class_name, x_center, y_center, width, height, confidence, id_number = detection
        # Convert relative coordinates to absolute coordinates
        h, w, _ = image.shape
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Put text including ID number
        cv2.putText(image, f'{id_number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Save image with bounding boxes
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

# Function to process a single JPEG file
def process_image(file_name):
    jpeg_path = os.path.join(jpeg_folder, file_name)
    txt_path = os.path.join(txt_folder, os.path.splitext(file_name)[0] + '.txt')
    if os.path.exists(txt_path):
        detections = read_detections(txt_path)
        draw_boxes(jpeg_path, detections, output_folder)
    else:
        # If no corresponding text file found, just copy the JPEG file to output folder
        output_path = os.path.join(output_folder, file_name)
        image = cv2.imread(jpeg_path)
        cv2.imwrite(output_path, image)

# Input folders
jpeg_folder = 'DCIM-drone1_drone1vid1'
txt_folder = '/home/jean-pierre/Desktop/26feb/drone1_vid1/interpolated_detected'

# Output folder
output_folder = '/media/jean-pierre/PortableSSD/work/d1v1_detected'

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# List of JPEG files
jpeg_files = [file_name for file_name in os.listdir(jpeg_folder) if file_name.endswith('.jpg')]

# Number of processes
num_processes = 10

# Process JPEG files in parallel using multiprocessing pool
with Pool(num_processes) as pool:
    pool.map(process_image, jpeg_files)

print("Bounding boxes drawn and saved successfully.")
