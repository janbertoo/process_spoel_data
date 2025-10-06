import cv2
import os
import numpy as np

folder_to_check = '/home/jean-pierre/Desktop/8dec_weekend/vid1'


filepaths = []
for file in os.listdir(folder_to_check):
	if file[-4:] == '.jpg':
		filepaths.append(os.path.join(folder_to_check,file))

filepaths.sort()

n_white_pix_memory = 0


for file in filepaths:
	#print(file)
	image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	img = image[760:3241, 59:5976]#    y:y+h, x:x+w]
	#n_white_pix = np.sum(img > 250)
	n_white_pix = np.sum(img == 255)
	difference = np.abs( n_white_pix - n_white_pix_memory )
	n_white_pix_memory = n_white_pix
	#print(difference)
	if difference > 10000:
		print(file)
		print(difference)

	#cv2.imwrite(file.replace('.jpg','_cropped.jpg'),img)