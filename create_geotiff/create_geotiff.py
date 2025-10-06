import os
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import pickle

#define base coordinates
Xbase = 2803987
Ybase = 1175193

image1path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone1_drone1vid2_frame01440.jpg'
image2path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone6_drone6vid2_frame01440.jpg'
image3path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone16_drone16vid2_frame00720.jpg'

hom1path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone1_drone1vid2_frame01440_hom.p'
hom2path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone6_drone6vid2_frame01440_hom.p'
hom3path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone16_drone16vid2_frame00720_hom.p'

with open(hom1path, 'rb') as pickle_f:
	hom1 = pickle.load(pickle_f)

with open(hom2path, 'rb') as pickle_f:
	hom2 = pickle.load(pickle_f)

with open(hom3path, 'rb') as pickle_f:
	hom3 = pickle.load(pickle_f)

# Load your images and homographies
image_paths = [image1path, image2path, image3path]
homographies = [hom1, hom2, hom3]

# Specify output GeoTIFF file path
output_path = "output_geotiff.tif"

# Open one image to get dimensions and transform information
src_image = cv2.imread(image_paths[0])
height, width, _ = src_image.shape

# Create a destination image to which all images will be warped
dst_image = np.zeros((height, width, 3), dtype=np.uint8)

# Warp each image using its homography and add to the destination image
for i in range(len(image_paths)):
    src_image = cv2.imread(image_paths[i])
    warped_image = cv2.warpPerspective(src_image, homographies[i], (width, height))
    warped_image_flipped = cv2.flip(warped_image, 0)
    dst_image = cv2.add(dst_image, warped_image_flipped)


min_x = 0#Xbase #2804000-Xbase
max_x = 2804190-Xbase

pixel_size_x = max_x - min_x
pixel_size_x = 1

min_y = 0 #1175215-Ybase
max_y = 2160 #Ybase +80 #1175295-Ybase

pixel_size_y = max_y - min_y
pixel_size_y = 1



# Define the output GeoTIFF metadata
transform = from_origin(min_x, max_y, pixel_size_x, pixel_size_y)
meta = {
    'driver': 'GTiff',
    'count': 3,  # Assuming 3 bands (RGB)
    'dtype': 'uint8',
    'width': width,
    'height': height,
    #'crs': 'EPSG:21781',#'EPSG:4326',  # You may need to adjust the CRS accordingly
    'crs': 'EPSG:4326',
    'transform': transform,
}

# Write the combined image to a GeoTIFF file
with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(dst_image.transpose(2, 0, 1))