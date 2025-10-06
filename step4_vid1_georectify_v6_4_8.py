import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import LineString
import datetime
import matplotlib
import gc
import pickle

matplotlib.use('Agg')

hom_diff_threshold = 0.4

curdir = os.getcwd()
divbyframes = 1

vidnumber_name = '4_8'
vidnumber = 4
mainpath = os.path.join(curdir,'cut_data')

#drone1frame_drone6frame_drone16frame = [49,0,328]   #vid1
#drone1frame_drone6frame_drone16frame = [60,0,211]   #vid2
#drone1frame_drone6frame_drone16frame = [30,0,115]   #vid3
drone1frame_drone6frame_drone16frame = [24,0,22]    #vid4
#drone1frame_drone6frame_drone16frame = [0,91,8]     #vid5

outputdir = os.path.join(curdir,'vid'+(vidnumber_name))
#outputdir = os.path.join('/home/jean-pierre/Desktop/8dec','vid'+str(vidnumber))
fps = 24
freq = 4
freq_old = 2

x=3840
y=2160

#define base coordinates
Xbase = 2803987
Ybase = 1175193

divby = 1
altcor = 0
slopecor = 1
#name_extension = '_divby'+str(divby)+'_altcor'+str(altcor)+'_slopecor'+str(slopecor)+'jpg'

#define water level to project on
waterlevel = 1489.485 #+ altcor

folder_with_gcps = 'labels_with_gcps'
folder_with_gcps_alternative = 'labels_with_gcps_no2percent'

#gcp 6 and 9
points_line1 = ((2804101.6-Xbase,1175278.832-Ybase),(2804107.518-Xbase,1175241.001-Ybase))

x1_line1 = points_line1[0][0]
y1_line1 = points_line1[0][1]
x2_line1 = points_line1[1][0]
y2_line1 = points_line1[1][1]

#print(x1_line1)
#print(y1_line1)
#print(y2_line1)
#print(y2_line1)
m_line1 = (y1_line1-y2_line1)/(x1_line1-x2_line1)                           #slope
b_line1 = (x1_line1*y2_line1 - x2_line1*y1_line1)/(x1_line1-x2_line1)       #y-intercept
#print(m_line1)
#print(b_line1)
#y_line1 = m_line1 * x_line1 + b_line1




#gcp 4 and 10
points_line2 = ((2804036.78-Xbase,1175271.824-Ybase),(2804069.799-Xbase,1175236.847-Ybase))

x1_line2 = points_line2[0][0]
y1_line2 = points_line2[0][1]
x2_line2 = points_line2[1][0]
y2_line2 = points_line2[1][1]

m_line2 = (y1_line2-y2_line2)/(x1_line2-x2_line2)
b_line2 = (x1_line2*y2_line2 - x2_line2*y1_line2)/(x1_line2-x2_line2)

#y_line2 = m_line2 * x_line2 + b_line2




#V2 Python (A0)
#Define Intrinsic (K) Matrix
#camera_matrix = np.array([[ 2.33387091e+03, 0.00000000e+00, 1.96778320e+03], [ 0.00000000e+00, 2.33680075e+03, 1.45223124e+03], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)
#Define distortion coefficients
#distortion_coefficients = np.array([[ 0.00329858, -0.01812266, -0.0012462, 0.00011494, 0.01102474 ]],np.float32)
#distortion_coefficients = np.array([[ k1,k2,p1,p2,k3 ]],np.float32)


f   = 3689.35702
cx  = -8.25948
cy  = -53.7588

k1  = -0.020482
k2  = -0.0531381
k3  = 0.110348
k4  = 0
p1  = -0.00111135
p2  = -0.00111865
b1  = -23.7142
b2  = 2.41531

fy = f
fx = f + b1

#V2 Python (A0)
#Define Intrinsic (K) Matrix
camera_matrix_drone1 = np.array([[ fx, 0.00000000e+00, cx ], [ 0.00000000e+00, fy, cy], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)
#camera_matrix_drone1 = np.array([[ 1e+03, 0.00000000e+00, 1e+03], [ 0.00000000e+00, 1e+03, 1e+03], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)

#Define distortion coefficients
distortion_coefficients_drone1 = np.array([[ 0,0,0,0,0 ]],np.float32)

f   = 3006.79553
cx  = 13.033
cy  = 12.3559

k1  = 0.00618627
k2  = -0.126217
k3  = 0.334084
k4  = -0.291903
p1  = -0.000269639
p2  = -0.000382
b1  = -20.2597
b2  = -0.5948

fy = f
fx = f + b1

#V2 Python (A0)
#Define Intrinsic (K) Matrix
camera_matrix_drone6 = np.array([[ fx, 0.00000000e+00, cx ], [ 0.00000000e+00, fy, cy], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)
#camera_matrix_drone6 = np.array([[ 1e+03, 0.00000000e+00, 1e+03], [ 0.00000000e+00, 1e+03, 1e+03], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)

#Define distortion coefficients
distortion_coefficients_drone6 = np.array([[ 0,0,0,0,0  ]],np.float32)

f   = 2729.98766
cx  = -33.8125
cy  = -12.1123

k1  = -0.0123651
k2  = 0.000249985
k3  = 0.0104065
k4  = 0
p1  = -0.00385929
p2  = -0.00262952
b1  = -17.5783
b2  = -0.064806

fy = f
fx = f + b1

#from manual calibration = shit
#fy = 5813.13017
#fx = 5170.03171
#cx = 2684.63643
#cy = 1271.12135

#V2 Python (A0)
#Define Intrinsic (K) Matrix
camera_matrix_drone16 = np.array([[ fx, 0.00000000e+00, cx ], [ 0.00000000e+00, fy, cy], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)
#camera_matrix_drone16 = np.array([[ 1e+03, 0.00000000e+00, 1e+03], [ 0.00000000e+00, 1e+03, 1e+03], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)

#Define distortion coefficients
distortion_coefficients_drone16 = np.array([[ k1,k2,p1,p2,k3 ]],np.float32)
distortion_coefficients_drone16 = np.array([[ 0,0,0,0,0 ]],np.float32)

vid1drone1 = ['DCIM-drone1_drone1vid'+str(vidnumber)]
vid1drone6 = ['DCIM-drone6_drone6vid'+str(vidnumber)]
vid1drone16 = ['DCIM-drone16_drone16vid'+str(vidnumber)]


opacity=1

def get_corrected_zcoor(xcoor,ycoor):

    lines_along_river = [(2804090.912822751,1175252.3458070147),(2804173,1175273)]
    line = LineString([lines_along_river[0], lines_along_river[1]])

    waterlevel_slope = -0.01437202672278633

    point = Point(xcoor,ycoor)
    x = np.array(point.coords[0])

    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)

    point_to_project = u + n*np.dot(x - u, n)
    
    if point_to_project[0] < lines_along_river[0][0]:
        distance = - np.sqrt( ( point_to_project[0] - lines_along_river[0][0] ) ** 2 + ( point_to_project[1] - lines_along_river[0][1] ) ** 2 )
    else:
        distance = np.sqrt( ( point_to_project[0] - lines_along_river[0][0] ) ** 2 + ( point_to_project[1] - lines_along_river[0][1] ) ** 2 )

    correction = -distance * waterlevel_slope

    return(correction)

def create_two_virtual_gcps(GCPS):

    virGCP1pix = ( (GCPS[0][0][0]+GCPS[2][0][0])/2,(GCPS[0][0][1]+GCPS[2][0][1])/2 )
    virGCP2pix = ( (GCPS[1][0][0]+GCPS[3][0][0])/2,(GCPS[1][0][1]+GCPS[3][0][1])/2 )

    virGCP1rw = ( (GCPS[0][1][0]+GCPS[2][1][0])/2,(GCPS[0][1][1]+GCPS[2][1][1])/2,(GCPS[0][1][2]+GCPS[2][1][2])/2 )
    virGCP2rw = ( (GCPS[1][1][0]+GCPS[3][1][0])/2,(GCPS[1][1][1]+GCPS[3][1][1])/2,(GCPS[1][1][2]+GCPS[3][1][2])/2 )

    vircoors = []

    vircoors.append([virGCP1pix,virGCP1rw])
    vircoors.append([virGCP2pix,virGCP2rw])

    #return([(virGCP1pix,virGCP1rw),(virGCP2pix,virGCP2rw)])
    return(vircoors)

def create_two_virtual_gcps_drone1_drone6_inw_ater(GCPS):

    virGCP1pix = ( (GCPS[0][0][0]+GCPS[1][0][0])/2,(GCPS[0][0][1]+GCPS[1][0][1])/2 )
    virGCP2pix = ( (GCPS[2][0][0]+GCPS[3][0][0])/2,(GCPS[2][0][1]+GCPS[3][0][1])/2 )

    virGCP1rw = ( (GCPS[0][1][0]+GCPS[1][1][0])/2,(GCPS[0][1][1]+GCPS[1][1][1])/2,(GCPS[0][1][2]+GCPS[1][1][2])/2 )
    virGCP2rw = ( (GCPS[2][1][0]+GCPS[3][1][0])/2,(GCPS[2][1][1]+GCPS[3][1][1])/2,(GCPS[2][1][2]+GCPS[3][1][2])/2 )

    vircoors = []

    vircoors.append([virGCP1pix,virGCP1rw])
    vircoors.append([virGCP2pix,virGCP2rw])

    #return([(virGCP1pix,virGCP1rw),(virGCP2pix,virGCP2rw)])
    return(vircoors)

def find_homography(UV, XYZ, K, distortion=np.zeros((1,4)), z=0):
    '''Find homography based on ground control points

    Parameters
    ----------
    UV : np.ndarray
        Nx2 array of image coordinates of gcp's
    XYZ : np.ndarray
        Nx3 array of real-world coordinates of gcp's
    K : np.ndarray
        3x3 array containing camera matrix
    distortion : np.ndarray, optional
        1xP array with distortion coefficients with P = 4, 5 or 8
    z : float, optional
        Real-world elevation on which the image should be projected

    Returns
    -------
    np.ndarray
        3x3 homography matrix

    Notes
    -----
    Function uses the OpenCV image rectification workflow as described in
    http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    starting with solvePnP.

    Examples
    --------
    >>> camera_id = 4
    >>> r = argus2.rest.get_rectification_data('kijkduin')
    >>> H = flamingo.rectification.find_homography(r[camera_id]['UV'],
                                                   r[camera_id]['XYZ'],
                                                   r[camera_id]['K'])
    '''

    UV = np.asarray(UV).astype(np.float32)
    XYZ = np.asarray(XYZ).astype(np.float32)
    K = np.asarray(K).astype(np.float32)
    
    # compute camera pose
    rvec, tvec = cv2.solvePnP(XYZ, UV, K, distortion)[-2:]
    
    # convert rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0]
    
    # assume height of projection plane
    R[:,2] = R[:,2] * z

    # add translation vector
    R[:,2] = R[:,2] + tvec.flatten()

    # compute homography
    H = np.linalg.inv(np.dot(K, R))

    # normalize homography
    H = H / H[-1,-1]

    cameraPosition = -np.matrix(R).T * np.matrix(tvec)
    #print(cameraPosition)

    return H

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

def _construct_rgba_vector(img, n_alpha=0):
    '''Construct RGBA vector to be used to color faces of pcolormesh

    Parameters
    ----------
    img : np.ndarray
        NxMx3 RGB image matrix
    n_alpha : int
        Number of border pixels to use to increase alpha

    Returns
    -------
    np.ndarray
        (N*M)x4 RGBA image vector
    '''

    alpha = np.ones(img.shape[:2])    
    
    if n_alpha > 0:
        for i, a in enumerate(np.linspace(0, 1, n_alpha)):
            alpha[:,[i,-2-i]] = a
        
    rgb = img[:,:,:].reshape((-1,3)) # we have 1 less faces than grid cells
    rgba = np.concatenate((rgb, alpha[:,:].reshape((-1, 1))), axis=1)

    if np.any(img > 1):
        rgba[:,:3] /= 255.0
    
    return rgba

def getlistofalldroneimages(mainpath,drone_vid_list):
    alldroneimages = []

    for folder in drone_vid_list:
        imagespath = os.path.join(mainpath,folder)
        for imfile in os.listdir(imagespath):
            if imfile[-4:] == '.jpg' and int(imfile[-9:-4]) % (fps/freq) == 0:
                alldroneimages.append(os.path.join(imagespath,imfile))

    alldroneimages.sort()

    return(alldroneimages)

alldrone_1_images = getlistofalldroneimages(mainpath,vid1drone1)
alldrone_6_images =  getlistofalldroneimages(mainpath,vid1drone6)
alldrone_16_images =  getlistofalldroneimages(mainpath,vid1drone16)

stagered_drone_ims = []

for i in range(0,20000):

    if i-drone1frame_drone6frame_drone16frame[0] < 0:
        drone1im = None
    else:
        try:
            drone1im = alldrone_1_images[i-drone1frame_drone6frame_drone16frame[0]]
        except:
            drone1im = None
    #print(drone1im)
    if i-drone1frame_drone6frame_drone16frame[1] < 0:
        drone6im = None
    else:
        try:
            drone6im = alldrone_6_images[i-drone1frame_drone6frame_drone16frame[1]]
        except:
            drone6im = None
    #print(drone6im)
    if i-drone1frame_drone6frame_drone16frame[2] < 0:
        drone16im = None
    else:
        try:
            drone16im = alldrone_16_images[i-drone1frame_drone6frame_drone16frame[2]]
        except:
            drone16im = None
    #print(drone16im)
    if drone1im == None and drone6im == None and drone16im == None:
        break

    stagered_drone_ims.append([drone1im,drone6im,drone16im])

#print(stagered_drone_ims)

stagered_drone_txts = []

stagered_drone_txts = []

stagered_drone_txts_alternative = []

for i in range(len(stagered_drone_ims)):
    try:
        jpg = stagered_drone_ims[i][0]
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps,(jpg_split[-1]).replace('.jpg','.txt'))
        drone1_txt = txt

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps_alternative,(jpg_split[-1]).replace('.jpg','.txt'))
        drone1_txt_alternative = txt
    except:
        drone1_txt = None
        drone1_txt_alternative = None
        
    try:
        jpg = stagered_drone_ims[i][1]
        #print(jpg)
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')
        #print(dcim_split)

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps,(jpg_split[-1]).replace('.jpg','.txt'))
        #print(txt)
        drone6_txt = txt
        
        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps_alternative,(jpg_split[-1]).replace('.jpg','.txt'))
        drone6_txt_alternative = txt
    except:
        drone6_txt = None
        drone6_txt_alternative = None
    
    try:
        jpg = stagered_drone_ims[i][2]
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps,(jpg_split[-1]).replace('.jpg','.txt'))
        drone16_txt = txt

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps_alternative,(jpg_split[-1]).replace('.jpg','.txt'))
        drone16_txt_alternative = txt
    except:
        drone16_txt = None
        drone16_txt_alternative = None
    
    stagered_drone_txts.append([drone1_txt,drone6_txt,drone16_txt])
    stagered_drone_txts_alternative.append([drone1_txt_alternative,drone6_txt_alternative,drone16_txt_alternative])

#print(stagered_drone_txts[0])

def get_pixel_and_rw_coords(txt_file):
    gcps_pixel_and_coors = []
    gcps_pixel_and_coors_squared = []
    gcps_pixel_and_coors_squared_opp = []
    print(txt_file)
    with open(txt_file, "r") as txt_file:
        lines_txt_file = txt_file.readlines()

        for line in lines_txt_file:
            splitted = line.split(', ')
            pixelcoor = float(splitted[0].split(' ')[0]),float(splitted[0].split(' ')[1])
            #if splitted[1].split(' ')[2][-2:] == '\n':
            #rwcoor = float(splitted[1].split(' ')[0])-Xbase,float(splitted[1].split(' ')[1])-Ybase,(float(splitted[1].split(' ')[2][:-2])-waterlevel)/divby
            #else:
            #print(( float(splitted[1].split(' ')[2]) - waterlevel -   get_corrected_zcoor( float(splitted[1].split(' ')[0]) , float(splitted[1].split(' ')[1]) ) ))
            #print(( get_corrected_zcoor( float(splitted[1].split(' ')[0]) , float(splitted[1].split(' ')[1]) ) ))
            rwcoor = float(splitted[1].split(' ')[0])-Xbase,float(splitted[1].split(' ')[1])-Ybase,   ( float(splitted[1].split(' ')[2]) - waterlevel -   get_corrected_zcoor( float(splitted[1].split(' ')[0]) , float(splitted[1].split(' ')[1]) ) )     /divby
            gcps_pixel_and_coors.append([pixelcoor,rwcoor])

            squared_pix = (pixelcoor[0]/x) ** 2 + (pixelcoor[1]/y) ** 2
            squared_pix_opp = ((x-pixelcoor[0])/x) ** 2 + (pixelcoor[1]/y) ** 2

            gcps_pixel_and_coors_squared.append([squared_pix,pixelcoor,rwcoor])
            gcps_pixel_and_coors_squared_opp.append([squared_pix_opp,pixelcoor,rwcoor])
    #print('')
    gcps_pixel_and_coors_squared.sort()
    #for entry in gcps_pixel_and_coors_squared:
    #    print(entry)
    gcps_pixel_and_coors_squared_opp.sort()
    #for entry in gcps_pixel_and_coors_squared_opp:
    #    print(entry)
    #print('')

    gcps_pixel_and_coors_first_4_in_corners = [gcps_pixel_and_coors_squared[0],gcps_pixel_and_coors_squared[-1],gcps_pixel_and_coors_squared_opp[0],gcps_pixel_and_coors_squared_opp[-1]]
    
    gcps_pixel_and_coors_first_4_in_corners = [[gcps_pixel_and_coors_squared[0][1],gcps_pixel_and_coors_squared[0][2]],
        [gcps_pixel_and_coors_squared[-1][1],gcps_pixel_and_coors_squared[-1][2]],
        [gcps_pixel_and_coors_squared_opp[0][1],gcps_pixel_and_coors_squared_opp[0][2]],
        [gcps_pixel_and_coors_squared_opp[-1][1],gcps_pixel_and_coors_squared_opp[-1][2]]
        ]

    return(gcps_pixel_and_coors,gcps_pixel_and_coors_first_4_in_corners)


names_georeferenced_image = []
countframe = 0
for i in range(len(stagered_drone_ims)):
    name = os.path.join(outputdir,'frame'+str(countframe).zfill(5)+'.jpg')
    countframe += 1
    names_georeferenced_image.append(name)

COUNT_SAMPLES = 0
timeold = datetime.datetime.now()
for i in range(len(stagered_drone_ims)):
    
    hom_drone1_flag = False
    hom_drone2_flag = False
    hom_drone3_flag = False

    #print(datetime.datetime.now()-timeold)
    timeold = datetime.datetime.now()
    
    try:
        name_georeferenced_image = names_georeferenced_image[i]
        if os.path.exists(name_georeferenced_image) == True:
            continue
        #print(name_georeferenced_image)
        
        COUNT_SAMPLES += 1
        if COUNT_SAMPLES % divbyframes != 0:
            continue

        name_original_image = stagered_drone_ims[i][0]
        #print(name_original_image)
        name_original_image2 = stagered_drone_ims[i][1]
        #print(name_original_image2)
        name_original_image3 = stagered_drone_ims[i][2]
        #print(name_original_image3)

        txt1 = stagered_drone_txts[i][0]
        #print(txt1)
        txt2 = stagered_drone_txts[i][1]
        #print(txt2)
        txt3 = stagered_drone_txts[i][2]
        #print(txt3)

        txt1_alternative = stagered_drone_txts_alternative[i][0]
        #print(txt1)
        txt2_alternative = stagered_drone_txts_alternative[i][1]
        #print(txt2)
        txt3_alternative = stagered_drone_txts_alternative[i][2]
        #print(txt3)

        figsize = (60, 40)
        cmap = 'Greys'
        color = True
        n_alpha = 0

        fig, ax = plt.subplots(figsize=figsize)
        
        









        



        









        if name_original_image2 == None:
            print('no drone2 im')
            hom_drone2_flag = False
        else:
            drone_image2 = cv2.imread(name_original_image2)
            (coorstxt2,coors_4ctxt2) = get_pixel_and_rw_coords(txt2)

            (coorstxt2_alternative,coors_4ctxt2_alternative) = get_pixel_and_rw_coords(txt2_alternative)

            virtuals_coors_txt2 = create_two_virtual_gcps_drone1_drone6_inw_ater(coors_4ctxt2)
            virtuals_coors_txt2_alternative = create_two_virtual_gcps_drone1_drone6_inw_ater(coors_4ctxt2_alternative)
            #print('eenzame uv coor:',virtuals_coors_txt2[0][0])
            

            try:
                #try to find homography with all pixel and rw coordinates from the labels_with_gcps folder
                #directly add 2 virtual gcps on water, to stabelize homography

                xyzs = []
                for i in range(len(coorstxt2)):
                    xyzs.append(coorstxt2[i][1])

                #xyzs.append(virtuals_coors_txt2[0][1])
                #xyzs.append(virtuals_coors_txt2[1][1])

                #drone1
                XYZ02 =                      np.array(xyzs, dtype=np.float32)
                uvs = []
                for i in range(len(coorstxt2)):
                    uvs.append(coorstxt2[i][0])
                
                #uvs.append(virtuals_coors_txt2[0][0])
                #uvs.append(virtuals_coors_txt2[1][0])

                UV02 =                   np.array(uvs, dtype=np.float32)
                

                #drone6
                hom_drone2 = find_homography(UV02, XYZ02, camera_matrix_drone6, distortion_coefficients_drone6)
            except:
                print('add 1 virtual gcp to drone6')
                try:
                    xyzs = []
                    for i in range(len(coorstxt2)):
                        xyzs.append(coorstxt2[i][1])

                    xyzs.append(virtuals_coors_txt2[0][1])

                    #drone1
                    XYZ02 =                      np.array(xyzs, dtype=np.float32)
                    uvs = []
                    for i in range(len(coorstxt2)):
                        uvs.append(coorstxt2[i][0])

                    uvs.append(virtuals_coors_txt2[0][0])


                    UV02 =                   np.array(uvs, dtype=np.float32)
                    

                    #drone6
                    hom_drone2 = find_homography(UV02, XYZ02, camera_matrix_drone6, distortion_coefficients_drone6)
                    for coor in coorstxt2:
                        print(coor)
                except:
                    print('add 2 gcps to drone6')
                    xyzs = []
                    for i in range(len(coorstxt2)):
                        xyzs.append(coorstxt2[i][1])
                    for i in range(len(virtuals_coors_txt2)):
                        xyzs.append(virtuals_coors_txt2[i][1])

                    #drone1
                    XYZ02 =                      np.array(xyzs, dtype=np.float32)
                    
                    uvs = []
                    for i in range(len(coorstxt2)):
                        uvs.append(coorstxt2[i][0])
                    for i in range(len(virtuals_coors_txt2)):
                        uvs.append(virtuals_coors_txt2[i][0])

                    UV02 =                   np.array(uvs, dtype=np.float32)
                    

                    #drone6
                    hom_drone2 = find_homography(UV02, XYZ02, camera_matrix_drone6, distortion_coefficients_drone6)

            


            hom_drone2_flag = True
            

            
            try:
                totaldiff = 0
                for q in range(len(hom_drone2)):
                    #print(q)
                    for w in range(len(hom_drone2[0])):
                        #print(w)
                        #print(hom_drone2[q][w])
                        #print(hom_drone2_remember[q][w])
                        totaldiff = totaldiff + np.abs( hom_drone2[q][w] - hom_drone2_remember[q][w] )
                print(totaldiff)
            except:
                print('no hom_drone2')

            if totaldiff > hom_diff_threshold:
                print('BAD HOMOGPRAPHY DRONE 2')
                
                xyzs = []
                for i in range(len(coorstxt2_alternative)):
                    xyzs.append(coorstxt2_alternative[i][1])
                #for i in range(len(virtuals_coors_txt2_alternative)):
                #    xyzs.append(virtuals_coors_txt2_alternative[i][1])
                xyzs.append(virtuals_coors_txt2[0][1])

                #drone1
                XYZ02 =                      np.array(xyzs, dtype=np.float32)
                
                uvs = []
                for i in range(len(coorstxt2_alternative)):
                    uvs.append(coorstxt2_alternative[i][0])
                #for i in range(len(virtuals_coors_txt2_alternative)):
                #    uvs.append(virtuals_coors_txt2_alternative[i][0])
                uvs.append(virtuals_coors_txt2[0][0])

                UV02 =                   np.array(uvs, dtype=np.float32)
                

                #drone6
                hom_drone2 = find_homography(UV02, XYZ02, camera_matrix_drone6, distortion_coefficients_drone6)

            else:
                hom_drone2_remember = hom_drone2

            
            X, Y = rectify_image(drone_image2, hom_drone2)
            bgrimg = drone_image2
            img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

            im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

            if color:
                rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
                im.set_array(None)
                im.set_edgecolor('none')
                im.set_facecolor(rgba)
                im.set_alpha(opacity)


















        if name_original_image == None:
            print('no drone1 im')
            hom_drone1_flag = False
        else:
            drone_image = cv2.imread(name_original_image)
            (coorstxt1,coors_4ctxt1) = get_pixel_and_rw_coords(txt1)
            virtuals_coors_txt1 = create_two_virtual_gcps_drone1_drone6_inw_ater(coors_4ctxt1)

            (coorstxt1_alternative,coors_4ctxt1_alternative) = get_pixel_and_rw_coords(txt1_alternative)
            virtuals_coors_txt1_alternative = create_two_virtual_gcps_drone1_drone6_inw_ater(coors_4ctxt1_alternative)

            try:
                xyzs = []
                for i in range(len(coorstxt1)):
                    xyzs.append(coorstxt1[i][1])

                #xyzs.append(virtuals_coors_txt2[0][1])
                #xyzs.append(virtuals_coors_txt2[1][1])

                #drone1
                XYZ0 =                      np.array(xyzs, dtype=np.float32)
                uvs = []
                for i in range(len(coorstxt1)):
                    uvs.append(coorstxt1[i][0])

                #uvs.append(virtuals_coors_txt2[0][0])
                #uvs.append(virtuals_coors_txt2[1][0])

                UV0 =                   np.array(uvs, dtype=np.float32)

                #drone1
                hom_drone1 = find_homography(UV0, XYZ0, camera_matrix_drone1, distortion_coefficients_drone1)
            except:
                xyzs = []
                for i in range(len(coorstxt1)):
                    xyzs.append(coorstxt1[i][1])
                for i in range(len(virtuals_coors_txt1)):
                    xyzs.append(virtuals_coors_txt1[i][1])

                #drone1
                XYZ0 =                      np.array(xyzs, dtype=np.float32)
                uvs = []
                for i in range(len(coorstxt1)):
                    uvs.append(coorstxt1[i][0])
                for i in range(len(virtuals_coors_txt1)):
                    uvs.append(virtuals_coors_txt1[i][0])

                UV0 =                   np.array(uvs, dtype=np.float32)

                #drone1
                hom_drone1 = find_homography(UV0, XYZ0, camera_matrix_drone1, distortion_coefficients_drone1)

            

            hom_drone1_flag = True
            
            try:
                totaldiff = 0
                for q in range(len(hom_drone1)):
                    for w in range(len(hom_drone1[0])):
                        totaldiff = totaldiff + np.abs( hom_drone1[q][w] - hom_drone1_remember[q][w] )
                print(totaldiff)
            except:
                print('no hom_drone1')

            if totaldiff > hom_diff_threshold:
                print('BAD HOMOGPRAPHY DRONE 1')
                
                xyzs = []
                for i in range(len(coorstxt1_alternative)):
                    xyzs.append(coorstxt1_alternative[i][1])
                for i in range(len(virtuals_coors_txt1_alternative)):
                    xyzs.append(virtuals_coors_txt1_alternative[i][1])

                #drone1
                XYZ02 =                      np.array(xyzs, dtype=np.float32)
                
                uvs = []
                for i in range(len(coorstxt1_alternative)):
                    uvs.append(coorstxt1_alternative[i][0])
                for i in range(len(virtuals_coors_txt1_alternative)):
                    uvs.append(virtuals_coors_txt1_alternative[i][0])

                UV02 =                   np.array(uvs, dtype=np.float32)
                

                #drone6
                hom_drone1 = find_homography(UV02, XYZ02, camera_matrix_drone6, distortion_coefficients_drone6)

            else:
                hom_drone1_remember = hom_drone1











            X, Y = rectify_image(drone_image, hom_drone1)

            #hom_drone1_remember = hom_drone1

            bgrimg = drone_image
            img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

            mask = np.zeros(img.shape[:2], dtype="uint8")

            for i in range(len(X)):
                for j in range(len(Y[0])):
                    if X[i][j] * m_line1 + b_line1 < Y[i][j]:
                        mask[i][j] = 1

            

            im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

            if color:
                rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
                im.set_array(None)
                im.set_edgecolor('none')
                im.set_facecolor(rgba)
                im.set_alpha(mask)
                #im.set_alpha(opacity)
















        if name_original_image3 == None:
            print('no drone3 im')
            hom_drone3_flag = False

        else:

            drone_image3 = cv2.imread(name_original_image3)
            (coorstxt3,coors_4ctxt3) = get_pixel_and_rw_coords(txt3)
            virtuals_coors_txt3 = create_two_virtual_gcps(coors_4ctxt3)

            (coorstxt3_alternative,coors_4ctxt3_alternative) = get_pixel_and_rw_coords(txt3_alternative)
            virtuals_coors_txt3_alternative = create_two_virtual_gcps(coors_4ctxt3_alternative)


            try:
                xyzs = []
                for i in range(len(coorstxt3)):
                    xyzs.append(coorstxt3[i][1])

                #drone1
                XYZ03 =                      np.array(xyzs, dtype=np.float32)
                uvs = []
                for i in range(len(coorstxt3)):
                    uvs.append(coorstxt3[i][0])
                UV03 =                   np.array(uvs, dtype=np.float32)
                #print(coors_4ctxt3)

                #drone16
                hom_drone3 = find_homography(UV03, XYZ03, camera_matrix_drone16, distortion_coefficients_drone16)
            except:
                xyzs = []
                for i in range(len(coorstxt3)):
                    xyzs.append(coorstxt3[i][1])
                for i in range(len(virtuals_coors_txt3)):
                    xyzs.append(virtuals_coors_txt3[i][1])

                #drone1
                XYZ03 =                      np.array(xyzs, dtype=np.float32)
                uvs = []
                for i in range(len(coorstxt3)):
                    uvs.append(coorstxt3[i][0])
                for i in range(len(virtuals_coors_txt3)):
                    uvs.append(virtuals_coors_txt3[i][0])

                UV03 =                   np.array(uvs, dtype=np.float32)
                #print(coors_4ctxt3)

                #drone16
                hom_drone3 = find_homography(UV03, XYZ03, camera_matrix_drone16, distortion_coefficients_drone16)

            

            hom_drone3_flag = True

            try:
                totaldiff = 0
                for q in range(len(hom_drone3)):
                    #print(q)
                    for w in range(len(hom_drone3[0])):
                        #print(w)
                        #print(hom_drone2[q][w])
                        #print(hom_drone2_remember[q][w])
                        totaldiff = totaldiff + np.abs( hom_drone3[q][w] - hom_drone3_remember[q][w] )
                print(totaldiff)
            except:
                print('no hom_drone3')


            if totaldiff > hom_diff_threshold:
                print('BAD HOMOGPRAPHY DRONE 3')
                
                xyzs = []
                for i in range(len(coorstxt3_alternative)):
                    xyzs.append(coorstxt3_alternative[i][1])
                for i in range(len(virtuals_coors_txt3_alternative)):
                    xyzs.append(virtuals_coors_txt3_alternative[i][1])

                #drone1
                XYZ02 =                      np.array(xyzs, dtype=np.float32)
                
                uvs = []
                for i in range(len(coorstxt3_alternative)):
                    uvs.append(coorstxt3_alternative[i][0])
                for i in range(len(virtuals_coors_txt3_alternative)):
                    uvs.append(virtuals_coors_txt3_alternative[i][0])

                UV02 =                   np.array(uvs, dtype=np.float32)
                

                #drone6
                hom_drone3 = find_homography(UV02, XYZ02, camera_matrix_drone6, distortion_coefficients_drone6)

            else:
                hom_drone3_remember = hom_drone3


            X, Y = rectify_image(drone_image3, hom_drone3)

            #hom_drone3_remember = hom_drone3

            bgrimg = drone_image3
            img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)




            mask = np.zeros(img.shape[:2], dtype="uint8")

            for i in range(len(X)):
                for j in range(len(Y[0])):
                    if X[i][j] * m_line2 + b_line2 > Y[i][j]:
                        mask[i][j] = 1

            


            im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

            if color:
                rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
                im.set_array(None)
                im.set_edgecolor('none')
                im.set_facecolor(rgba)
                im.set_alpha(mask)
                #im.set_alpha(opacity)





        

    except:
        print('NIET GELUKT!!!!')
        X = None
        Y = None
        #hom_drone1 = False
        #hom_drone2 = False
        #hom_drone2 = False








    







    









    ax.set_aspect('equal')
        
    #plot red dots at the loction the GCPS are supposed to be
    #x1_gcps = np.array([GCPs[0][0],GCPs[1][0],GCPs[2][0],GCPs[3][0],GCPs[4][0],GCPs[5][0],GCPs[6][0],GCPs[7][0],GCPs[8][0],GCPs[9][0],GCPs[10][0],GCPs[11][0]])
    #y2_gcps = np.array([GCPs[0][1],GCPs[1][1],GCPs[2][1],GCPs[3][1],GCPs[4][1],GCPs[5][1],GCPs[6][1],GCPs[7][1],GCPs[8][1],GCPs[9][1],GCPs[10][1],GCPs[11][1]])

    #for i in range(len(x1_gcps)):
    #    plt.Circle(( x1_gcps[i] , y2_gcps[i] ), 1 )
    

    
    '''
    x1_gcps = []
    y2_gcps = []
    for i in range(len(coorstxt1)):
        #print((coorstxt1[i][1][0],coorstxt1[i][1][1]))
        #plt.Circle((coorstxt1[i][1][0],coorstxt1[i][1][1]), 10 )
        x1_gcps.append(coorstxt1[i][1][0])
        y2_gcps.append(coorstxt1[i][1][1])
    for i in range(len(coorstxt2)):
        #plt.Circle((coorstxt2[i][1][0],coorstxt2[i][1][1]), 10 )
        x1_gcps.append(coorstxt2[i][1][0])
        y2_gcps.append(coorstxt2[i][1][1])
    for i in range(len(coorstxt3)):
        #plt.Circle((coorstxt3[i][1][0],coorstxt3[i][1][1]), 10 )
        x1_gcps.append(coorstxt3[i][1][0])
        y2_gcps.append(coorstxt3[i][1][1])

    ax.scatter(x1_gcps,y2_gcps, marker='o',color='r')
    '''

    




    #fig, ax = plt.subplots()

    #ax.scatter(x1,y2, marker='o',color='r')
            
    #plt.ylim(1175210-Ybase, 1175310-Ybase)
    #plt.xlim(2803982-Xbase, 2804200-Xbase)

    plt.ylim(1175215-Ybase, 1175295-Ybase)
    plt.xlim(2804000-Xbase, 2804190-Xbase)
    #plt.ylim(210, 310)
    #plt.xlim(0, 200)
    plt.tight_layout()
    #plt.autoscale(enable=True, axis='y', tight=True)
    #plt.savefig(name_georeferenced_image.replace('.jpg',name_extension))
    #print(' ')
    print('saving '+name_georeferenced_image)
    plt.savefig(name_georeferenced_image)
    print(' ')

    
    if hom_drone1_flag != False:
        txt1_name = (txt1.split('/')[-1]).split('.txt')[0]
        file_1 = open(name_georeferenced_image.replace('.jpg','_'+txt1_name+'_hom.p'), 'wb')
        pickle.dump(hom_drone1,file_1)

    
    if hom_drone2_flag != False:
        txt2_name = (txt2.split('/')[-1]).split('.txt')[0]
        file_2 = open(name_georeferenced_image.replace('.jpg','_'+txt2_name+'_hom.p'), 'wb')
        pickle.dump(hom_drone2,file_2)

    
    if hom_drone3_flag != False:
        txt3_name = (txt3.split('/')[-1]).split('.txt')[0]
        file_3 = open(name_georeferenced_image.replace('.jpg','_'+txt3_name+'_hom.p'), 'wb')
        pickle.dump(hom_drone3,file_3)

    
    
    fig.clf()
    plt.close(fig)
    plt.close()
    plt.clf()
    del X, Y
    gc.collect()