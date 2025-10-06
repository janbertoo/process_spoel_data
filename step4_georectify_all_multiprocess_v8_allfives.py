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
import multiprocessing

matplotlib.use('Agg')

hom_diff_threshold = 0.4


# Set the number of processes you want to use (adjust as needed)
num_processes = 8

curdir = os.getcwd()
divbyframes = 1

mainpath = os.path.join(curdir,'cut_data')

drone1frame_drone6frame_drone16frames = [
    [49,0,328],     #vid1
    [60,0,211],     #vid2
    [30,0,115],     #vid3
    [24,0,22],      #vid4
    [0,91,1],        #vid5
]

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

#define water level to project on
waterlevel = 1489.485 #+ altcor

#gcp 6 and 9
points_line1 = ((2804101.6-Xbase,1175278.832-Ybase),(2804107.518-Xbase,1175241.001-Ybase))

x1_line1 = points_line1[0][0]
y1_line1 = points_line1[0][1]
x2_line1 = points_line1[1][0]
y2_line1 = points_line1[1][1]

m_line1 = (y1_line1-y2_line1)/(x1_line1-x2_line1)                           #slope
b_line1 = (x1_line1*y2_line1 - x2_line1*y1_line1)/(x1_line1-x2_line1)       #y-intercept

#gcp 4 and 10
points_line2 = ((2804036.78-Xbase,1175271.824-Ybase),(2804069.799-Xbase,1175236.847-Ybase))

x1_line2 = points_line2[0][0]
y1_line2 = points_line2[0][1]
x2_line2 = points_line2[1][0]
y2_line2 = points_line2[1][1]

m_line2 = (y1_line2-y2_line2)/(x1_line2-x2_line2)
b_line2 = (x1_line2*y2_line2 - x2_line2*y1_line2)/(x1_line2-x2_line2)



opacity=1

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

def getlistofalldroneimages(mainpath,drone_vid_list,vidnumber,dronenumber):
    alldroneimages = []

    for folder in drone_vid_list:
        imagespath = os.path.join(mainpath,folder)
        for imfile in os.listdir(imagespath):
            if imfile[-4:] == '.jpg' and int(imfile[-9:-4]) % (fps/freq) == 0:
                #print(imfile[-9:-4])
                #print(mainpath)
                hompath = os.path.join(mainpath,'DCIM-drone'+str(dronenumber),'drone'+str(dronenumber)+'vid'+str(vidnumber),'initial_homs')
                correct_hompath = os.path.join(mainpath,'DCIM-drone'+str(dronenumber),'drone'+str(dronenumber)+'vid'+str(vidnumber),'correct_homs')
                homfile = imfile.replace('.jpg','_hom.p')
                alldroneimages.append([os.path.join(imagespath,imfile),os.path.join(hompath,homfile)])

    alldroneimages.sort()

    return(alldroneimages)







stagered_drone_ims = []

for vidnumber in [1,2,3,4,5]:
    outputdir = os.path.join(curdir,'vid_'+str(vidnumber))
    #print(outputdir)
    drone1frame_drone6frame_drone16frame = drone1frame_drone6frame_drone16frames[vidnumber-1]
    #print(drone1frame_drone6frame_drone16frame)

    vid1drone1 = ['DCIM-drone1_drone1vid'+str(vidnumber)]
    vid1drone6 = ['DCIM-drone6_drone6vid'+str(vidnumber)]
    vid1drone16 = ['DCIM-drone16_drone16vid'+str(vidnumber)]

    alldrone_1_images_and_homs = getlistofalldroneimages(mainpath,vid1drone1,vidnumber,1)
    alldrone_6_images_and_homs =  getlistofalldroneimages(mainpath,vid1drone6,vidnumber,6)
    alldrone_16_images_and_homs =  getlistofalldroneimages(mainpath,vid1drone16,vidnumber,16)

    framenumber = 0
    for i in range(0,20000):
        framenumber = framenumber + 1

        if i-drone1frame_drone6frame_drone16frame[0] < 0:
            drone1im = None
            drone1hom = None
        else:
            try:
                drone1im = alldrone_1_images_and_homs[i-drone1frame_drone6frame_drone16frame[0]][0]
                drone1hom = alldrone_1_images_and_homs[i-drone1frame_drone6frame_drone16frame[0]][1]
            except:
                drone1im = None
                drone1hom = None
        #print(drone1im)
        if i-drone1frame_drone6frame_drone16frame[1] < 0:
            drone6im = None
            drone6hom = None
        else:
            try:
                drone6im = alldrone_6_images_and_homs[i-drone1frame_drone6frame_drone16frame[1]][0]
                drone6hom = alldrone_6_images_and_homs[i-drone1frame_drone6frame_drone16frame[1]][1]
            except:
                drone6im = None
                drone6hom = None
        #print(drone6im)
        if i-drone1frame_drone6frame_drone16frame[2] < 0:
            drone16im = None
            drone16hom = None
        else:
            try:
                drone16im = alldrone_16_images_and_homs[i-drone1frame_drone6frame_drone16frame[2]][0]
                drone16hom = alldrone_16_images_and_homs[i-drone1frame_drone6frame_drone16frame[2]][1]
            except:
                drone16im = None
                drone16hom = None
        #print(drone16im)
        if drone1im == None and drone6im == None and drone16im == None:
            break

        stagered_drone_ims.append([[drone1im,drone6im,drone16im],[drone1hom,drone6hom,drone16hom],os.path.join(outputdir,'frame'+str(framenumber).zfill(5)+'.jpg')])

def georectify_three_images(data):
    framenumber = int((data[2].split('/frame')[-1]).split('.jpg')[0])
    #if os.path.exists(outputpath) == True:
        #break
    if framenumber % 8 == 0:
        images = data[0]
        homographies = data[1]
        outputpath = data[2]

        im1 = images[0]
        im2 = images[1]
        im3 = images[2]

        hom1 = homographies[0]
        hom2 = homographies[1]
        hom3 = homographies[2]

        figsize = (60, 40)
        cmap = 'Greys'
        color = True
        n_alpha = 0

        fig, ax = plt.subplots(figsize=figsize)




        #DRONE 6
        if im2 == None:
            pointlessparameter = 1
            #print('no Drone 6 image')
        else:
            drone2_image = cv2.imread(im2)

            with open(hom2, 'rb') as pickle_f:
                drone2_homography = pickle.load(pickle_f)


            X, Y = rectify_image(drone2_image, drone2_homography)
            bgrimg = drone2_image
            img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

            im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

            if color:
                rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
                im.set_array(None)
                im.set_edgecolor('none')
                im.set_facecolor(rgba)
                im.set_alpha(opacity)



        #DRONE 1
        if im1 == None:
            pointlessparameter = 1
            #print('no Drone 1 image')
        else:
            drone1_image = cv2.imread(im1)

            with open(hom1, 'rb') as pickle_f:
                drone1_homography = pickle.load(pickle_f)

            X, Y = rectify_image(drone1_image, drone1_homography)
            bgrimg = drone1_image
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







        #DRONE 16
        if im3 == None:
            pointlessparameter = 1
            #print('no Drone 16 image')
        else:
            drone3_image = cv2.imread(im3)

            with open(hom3, 'rb') as pickle_f:
                drone3_homography = pickle.load(pickle_f)


            X, Y = rectify_image(drone3_image, drone3_homography)
            bgrimg = drone3_image
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


        ax.set_aspect('equal')



        plt.ylim(1175215-Ybase, 1175295-Ybase)
        plt.xlim(2804000-Xbase, 2804190-Xbase)
        plt.tight_layout()
        plt.savefig(data[2])

        fig.clf()
        plt.close(fig)
        plt.close()
        plt.clf()
        del X, Y
        gc.collect()
        print(data[2])



if __name__ == "__main__":
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the processing function to the list of SVG files
        #print(stagered_drone_ims)
        pool.map(georectify_three_images, stagered_drone_ims)