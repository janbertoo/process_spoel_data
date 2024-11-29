import os
import pickle
import cv2
import numpy as np
import math
import multiprocessing

num_processes = 8

curdir = os.getcwd()
#newdir = '/home/jean-pierre/Desktop/5feb/detectioncoordinates'

image_path = os.path.join(curdir,'cut_data/DCIM-drone1_drone1vid1/DCIM-drone1_drone1vid1_frame10140.jpg')

#print(image_path)
#read the image
image = cv2.imread(image_path)

#determine the height and width of the image
img_height = image.shape[0]
img_width = image.shape[1]



def rectify_image(img, H):
    '''Get projection of image pixels in real-world coordinates
       given an image and homography

    Parameters
    ----------
    img : np.ndarray
        NxMx1 or NxMx3 image matrix
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing real-world x-coordinates
    np.ndarray
        NxM matrix containing real-world y-coordinates
    '''

    U, V = get_pixel_coordinates(img)
    #for i in range(len(U)):
    #    print(U[i])
    #for i in range(len(V)):
    #    print(V[i])
    X, Y = rectify_coordinates(U, V, H)

    return X, Y

def get_pixel_coordinates(img):
    '''Get pixel coordinates given an image

    Parameters
    ----------
    img : np.ndarray
        NxMx1 or NxMx3 image matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing u-coordinates
    np.ndarray
        NxM matrix containing v-coordinates
    '''

    # get pixel coordinates
    U, V = np.meshgrid(range(img.shape[1]),
                       range(img.shape[0]))

    return U, V

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



def get_exact_location(U,V,X,Y):
	if U > img_width:
		U = img_width
	if U < 0:
		U = 0
	if V > img_height:
		V = img_height
	if V < 0:
		V = 0
	U_low = math.floor(U-1)
	#if U_low < 0:
	#	U_low = 0
	#if U_low > img_width: #index will fall out of range, so change the index
	#	U_low = img_width - 1
	U_high = math.ceil(U-1)
	#if U_high < 0:
	#	U_high = 0
	#if U_high > img_width: #index will fall out of range, so change the index
	#	U_high = img_width - 1

	V_low = math.floor(V-1)
	#if V_low < 0:
	#	V_low = 0
	#if V_low > img_height: #index will fall out of range, so change the index
	#	V_low = img_height - 1
	V_high = math.ceil(V-1)
	#if V_high < 0:
	#	V_high = 0
	#if V_high > img_height: #index will fall out of range, so change the index
	#	V_high = img_height - 1

	X_ll = X[V_low][U_low]
	X_lh = X[V_low][U_high]
	X_hl = X[V_high][U_low]
	X_hh = X[V_high][U_high]

	Y_ll = Y[V_low][U_low]
	Y_lh = Y[V_low][U_high]
	Y_hl = Y[V_high][U_low]
	Y_hh = Y[V_high][U_high]
	
	#print(' ')
	#print(X_ll,Y_ll)
	#print(X_lh,Y_lh)
	#print(X_hl,Y_hl)
	#print(X_hh,Y_hh)
	#print(' ')

	X_exact = 	0.5 * ( 1 - ( U - U_low ) ) * ( X_ll + X_hl ) / 2 + 0.5 * ( 1 - ( U_high - U ) ) * ( X_lh + X_hh ) / 2 + 0.5 * ( 1 - ( V - V_low ) ) * ( X_lh + X_ll ) / 2 + 0.5 * ( 1 - ( V_high - V ) ) * ( X_hl + X_hh ) / 2
	Y_exact = 	0.5 * ( 1 - ( U - U_low ) ) * ( Y_ll + Y_hl ) / 2 + 0.5 * ( 1 - ( U_high - U ) ) * ( Y_lh + Y_hh ) / 2 + 0.5 * ( 1 - ( V - V_low ) ) * ( Y_lh + Y_ll ) / 2 + 0.5 * ( 1 - ( V_high - V ) ) * ( Y_hl + Y_hh ) / 2


	return(X_exact,Y_exact)





drone_vid_all = []
for dronenumber in [1,6,16]:
	for vidnumber in [1,2,3,4,5]:
		drone_vid_all.append([dronenumber,vidnumber])

def convert_detections_into_coordinates(data):
	dronenumber = data[0]
	vidnumber = data[1]
	basepath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)

	all_homs_path = os.path.join(basepath,'correct_homs_18jan_all')
	all_inferred_path = os.path.join(basepath,'inferred_perdrone')

	new_all_inferred_coors_path = os.path.join(basepath,'inferred_perdrone_coordinates')
	#new_all_inferred_coors_path = os.path.join('/home/jean-pierre/Desktop/8feb','inferred_coordinates')


	if os.path.isdir(new_all_inferred_coors_path) == False:
		os.mkdir(new_all_inferred_coors_path)

	alltxts = []
	for file in os.listdir(all_inferred_path):
		if file[-4:] == '.txt':
			alltxts.append(file)

	alltxts.sort()

	#TL_U_pixel_smallest = 10
	#TR_U_pixel_largest = 3500

	for file in alltxts:
		homography_path 			= os.path.join(all_homs_path,file.replace('.txt','_hom.p'))
		inferred_path 				= os.path.join(all_inferred_path,file)
		inferred_coordinates_path 	= os.path.join(new_all_inferred_coors_path,file)
		#print(inferred_path)

		#read the homography
		with open(homography_path, 'rb') as pickle_f1:
			homography = pickle.load(pickle_f1)

		#read txt file
		with open(inferred_path) as f:
			lines = f.readlines()

		detections = []

		#read the line and gather the centerX, centerY, width and height of the detection
		for line in lines:
			splitted = line.split(' ')
			U = float(splitted[1])
			V = float(splitted[2])
			width = float(splitted[3])
			height = float(splitted[4])
			confidence = float(splitted[5])
			detections.append([U,V,width,height,confidence])




		#define the coordinates of all pixels in the image
		X,Y = rectify_image(image,homography)
		#print(X)
		#print(Y)

		with open(inferred_coordinates_path, 'w') as f:
			

			#now go through detections and determine pixel and XY coordinates
			for detection in detections:
				U_pixel = detection[0] * img_width
				V_pixel = img_height - ( detection[1] * img_height ) #this is because in cv2 the pixels start counting top left instead of bottom left
				width_pixel = detection[2] * img_width
				height_pixel = detection[3] * img_height
				#print(U_pixel)
				#print(V_pixel)
				confidence = detection[4]
				
				#define the top left (TL), top right (TR), bottom left (BL) and bottom right (BR) coordinates from the yolo detections
				TL_U_pixel = U_pixel - 0.5 * width_pixel
				TL_V_pixel = V_pixel + 0.5 * height_pixel

				TR_U_pixel = U_pixel + 0.5 * width_pixel
				TR_V_pixel = V_pixel + 0.5 * height_pixel
				
				BL_U_pixel = U_pixel - 0.5 * width_pixel
				BL_V_pixel = V_pixel - 0.5 * height_pixel
				
				BR_U_pixel = U_pixel + 0.5 * width_pixel
				BR_V_pixel = V_pixel - 0.5 * height_pixel

				#if TL_U_pixel < TL_U_pixel_smallest:
				#	TL_U_pixel_smallest = TL_U_pixel
				#	print(TL_U_pixel_smallest)

				#if TR_U_pixel > TR_U_pixel_largest:
				#	TR_U_pixel_largest = TR_U_pixel
				#	print(TR_U_pixel_largest)

				X_center, Y_center = get_exact_location(U_pixel,V_pixel,X,Y)

				X_TL, Y_TL = get_exact_location(TL_U_pixel, TL_V_pixel,X,Y)
				X_TR, Y_TR = get_exact_location(TR_U_pixel, TR_V_pixel,X,Y)
				X_BL, Y_BL = get_exact_location(BL_U_pixel, BL_V_pixel,X,Y)
				X_BR, Y_BR = get_exact_location(BR_U_pixel, BR_V_pixel,X,Y)
				#X_pixel = X[int(V_pixel)][int(U_pixel)]
				#Y_pixel = Y[int(V_pixel)][int(U_pixel)]

				f.write('('+str(X_center)+' '+str(Y_center)+') ')
				f.write('('+str(X_TL)+' '+str(Y_TL)+') ')
				f.write('('+str(X_TR)+' '+str(Y_TR)+') ')
				f.write('('+str(X_BL)+' '+str(Y_BL)+') ')
				f.write('('+str(X_BR)+' '+str(Y_BR)+') ')
				f.write(str(confidence)+'\n')

if __name__ == "__main__":
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the processing function to the list of SVG files
        #print(stagered_drone_ims)
        pool.map(convert_detections_into_coordinates, drone_vid_all)