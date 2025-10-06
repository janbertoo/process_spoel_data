import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

curdir = os.getcwd()

vidnumber = 4
mainpath = os.path.join(curdir,'cut_data')

outputdir = os.path.join(curdir,'vid'+str(vidnumber))
outputdir = os.path.join('/home/jean-pierre/Desktop/forVirginia','vid'+str(vidnumber))

fps = 24
freq = 4
freq_old = 2

x=3840
y=2160

#define base coordinates
Xbase = 2803987
Ybase = 1175193

divby = 10

#define water level to project on
waterlevel = 1489.485

#Define Intrinsic (K) Matrix
camera_matrix = np.array([[ 1e+03, 0.00000000e+00, 1e+03], [ 0.00000000e+00, 1e+03, 1e+03], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float32)
#Define distortion coefficients
distortion_coefficients = np.array([[ 0, 0, 0, 0, 0 ]],np.float32)

vid1drone1 = ['drone1vid'+str(vidnumber)]
vid1drone6 = ['done6vid'+str(vidnumber)]
vid1drone16 = ['drone16vid'+str(vidnumber)]

drone1frame_drone6frame_drone16frame = [48,0,329]




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
for i in range(len(stagered_drone_ims)):
    try:
        jpg = stagered_drone_ims[i][0]
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],'labels_with_gcps',(jpg_split[-1]).replace('.jpg','.txt'))
        drone1_txt = txt
    except:
        drone1_txt = None
        
    try:
        jpg = stagered_drone_ims[i][1]
        #print(jpg)
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')
        #print(dcim_split)

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],'labels_with_gcps',(jpg_split[-1]).replace('.jpg','.txt'))
        #print(txt)
        drone6_txt = txt
        #print(drone6_txt)
    except:
        drone6_txt = None
    
    try:
        jpg = stagered_drone_ims[i][2]
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],'labels_with_gcps',(jpg_split[-1]).replace('.jpg','.txt'))
        drone16_txt = txt
    except:
        drone16_txt = None
    
    stagered_drone_txts.append([drone1_txt,drone6_txt,drone16_txt])

print(stagered_drone_txts[0])

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
            rwcoor = float(splitted[1].split(' ')[0])-Xbase,float(splitted[1].split(' ')[1])-Ybase,(float(splitted[1].split(' ')[2])-waterlevel)/divby
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
for i in range(len(stagered_drone_ims)):
    name_georeferenced_image = names_georeferenced_image[i]
    if os.path.exists(name_georeferenced_image) == True:
        continue
    print(name_georeferenced_image)
    
    COUNT_SAMPLES += 1
    if COUNT_SAMPLES % 100 != 0:
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

    figsize = (60, 40)
    cmap = 'Greys'
    color = True
    n_alpha = 0

    fig, ax = plt.subplots(figsize=figsize)
    
    if name_original_image == None:
        print('no drone1 im')
    else:
        drone_image = cv2.imread(name_original_image)
        (coorstxt1,coors_4ctxt1) = get_pixel_and_rw_coords(txt1)

        #drone1
        XYZ0 =                      np.array([coors_4ctxt1[0][1],
                                            coors_4ctxt1[1][1],
                                            coors_4ctxt1[2][1],
                                            coors_4ctxt1[3][1],
                                            # Add more GCPs as needed
                                            ], dtype=np.float32)

        UV0 =                   np.array([coors_4ctxt1[0][0],
                                        coors_4ctxt1[1][0],
                                        coors_4ctxt1[2][0],
                                        coors_4ctxt1[3][0],
                                        # Add more GCPs as needed
                                        ], dtype=np.float32)






        #drone1
        hom = find_homography(UV0, XYZ0, camera_matrix, distortion_coefficients)
        X, Y = rectify_image(drone_image, hom)

        bgrimg = drone_image
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        

        im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

        if color:
            rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
            im.set_array(None)
            im.set_edgecolor('none')
            im.set_facecolor(rgba)



    if name_original_image2 == None:
        print('no drone2 im')
    else:
        drone_image2 = cv2.imread(name_original_image2)
        (coorstxt2,coors_4ctxt2) = get_pixel_and_rw_coords(txt2)

        #drone6
        XYZ02 =                      np.array([coors_4ctxt2[0][1],
                                            coors_4ctxt2[1][1],
                                            coors_4ctxt2[2][1],
                                            coors_4ctxt2[3][1],
                                            # Add more GCPs as needed
                                            ], dtype=np.float32)

        UV02 =                   np.array([coors_4ctxt2[0][0],
                                        coors_4ctxt2[1][0],
                                        coors_4ctxt2[2][0],
                                        coors_4ctxt2[3][0],
                                        # Add more GCPs as needed
                                        ], dtype=np.float32)

        #drone6
        hom = find_homography(UV02, XYZ02, camera_matrix, distortion_coefficients)
        X, Y = rectify_image(drone_image2, hom)

        bgrimg = drone_image2
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

        if color:
            rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
            im.set_array(None)
            im.set_edgecolor('none')
            im.set_facecolor(rgba)




    if name_original_image3 == None:
        print('no drone3 im')
    else:
        drone_image3 = cv2.imread(name_original_image3)
        (coorstxt3,coors_4ctxt3) = get_pixel_and_rw_coords(txt3)

        #drone16
        XYZ03 =                      np.array([coors_4ctxt3[0][1],
                                            coors_4ctxt3[1][1],
                                            coors_4ctxt3[2][1],
                                            coors_4ctxt3[3][1],
                                            #GCPs[2],
                                            #GCPs[1],
                                            # Add more GCPs as needed
                                            ], dtype=np.float32)
        #print(XYZ03)

        UV03 =                   np.array([coors_4ctxt3[0][0],
                                        coors_4ctxt3[1][0],
                                        coors_4ctxt3[2][0],
                                        coors_4ctxt3[3][0],
                                        #[2670,1549],
                                        #[3195,238],
                                        # Add more GCPs as needed
                                        ], dtype=np.float32)
        #print(UV03)
        print(coors_4ctxt3)

        #drone16
        hom = find_homography(UV03, XYZ03, camera_matrix, distortion_coefficients)
        X, Y = rectify_image(drone_image3, hom)

        bgrimg = drone_image3
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

        if color:
            rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
            im.set_array(None)
            im.set_edgecolor('none')
            im.set_facecolor(rgba)







    ax.set_aspect('equal')
        
    #plot red dots at the loction the GCPS are supposed to be
    #x1 = np.array([GCPs[0][0],GCPs[1][0],GCPs[2][0],GCPs[3][0],GCPs[4][0],GCPs[5][0],GCPs[6][0],GCPs[7][0],GCPs[8][0],GCPs[9][0],GCPs[10][0],GCPs[11][0]])
    #y2 = np.array([GCPs[0][1],GCPs[1][1],GCPs[2][1],GCPs[3][1],GCPs[4][1],GCPs[5][1],GCPs[6][1],GCPs[7][1],GCPs[8][1],GCPs[9][1],GCPs[10][1],GCPs[11][1]])

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
    plt.savefig(name_georeferenced_image)

        
    fig.clf()
    plt.close()
    del X, Y