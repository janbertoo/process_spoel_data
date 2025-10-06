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

curdir = os.getcwd()

outputdir = os.path.join(curdir,'checkhoms')
outputdir = '/home/jean-pierre/Desktop/5feb/checkhoms'

cut_data_dir = os.path.join(curdir,'cut_data')

homsdir = 'correct_homs_18jan'

#define base coordinates
Xbase = 2803987
Ybase = 1175193

num_processes = 8

line1 = [[46.3,74.4],[59.6,75.2]]
line2 = [[126.1,59.8],[77.1,48.0]]
line3 = [[114.1,78.5],[111.7,76.7]]

line4 = [[15.5,56.3],[20.3,58.7]]
line5 = [[17.4,75.0],[30.6,76.3]]
line6 = [[21.3,72.4],[29.9,73.2]]
line7 = [[73.6,44.6],[70.0,40.7]]
line8 = [[173.9,64.7],[172.3,68.1]]
line9 = [[185.8,81.1],[178.0,82.0]]
line10 = [[184.6,88.7],[176.8,83.9]]

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

def georectify(data):
    try:
        homfile = data[0]
        jpgfile = data[1]
        outhomfilepath = data[2]
        outfilepath = data[3]

        figsize = (60, 40)
        cmap = 'Greys'
        color = True
        n_alpha = 0

        fig, ax = plt.subplots(figsize=figsize)

        image = cv2.imread(jpgfile)

        with open(homfile, 'rb') as pickle_f1:
        	homography = pickle.load(pickle_f1)

        X, Y = rectify_image(image, homography)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

        rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
        im.set_array(None)
        im.set_edgecolor('none')
        im.set_facecolor(rgba)

        ax.set_aspect('equal')
        plt.ylim(1175215-Ybase, 1175295-Ybase)
        plt.xlim(2804000-Xbase, 2804190-Xbase)
        plt.plot((line1[0][0],line1[1][0]),(line1[0][1],line1[1][1]), linewidth=20, color='red')
        plt.plot((line2[0][0],line2[1][0]),(line2[0][1],line2[1][1]), linewidth=20, color='red')
        plt.plot((line3[0][0],line3[1][0]),(line3[0][1],line3[1][1]), linewidth=20, color='red')

        plt.plot((line4[0][0],line4[1][0]),(line4[0][1],line4[1][1]), linewidth=20, color='red')
        plt.plot((line5[0][0],line5[1][0]),(line5[0][1],line5[1][1]), linewidth=20, color='red')
        plt.plot((line6[0][0],line6[1][0]),(line6[0][1],line6[1][1]), linewidth=20, color='red')
        plt.plot((line7[0][0],line7[1][0]),(line7[0][1],line7[1][1]), linewidth=20, color='red')
        plt.plot((line8[0][0],line8[1][0]),(line8[0][1],line8[1][1]), linewidth=20, color='red')
        plt.plot((line9[0][0],line9[1][0]),(line9[0][1],line9[1][1]), linewidth=20, color='red')
        plt.plot((line10[0][0],line10[1][0]),(line10[0][1],line10[1][1]), linewidth=20, color='red')

        #plt.text(20, 20, str(data[2]), size='xx-large')
        plt.ylim(1175215-Ybase, 1175295-Ybase)
        plt.xlim(2804000-Xbase, 2804190-Xbase)
        plt.tight_layout()
        plt.savefig(outfilepath)

        file = open(outhomfilepath, 'wb')
        pickle.dump(homography,file)

        fig.clf()
        plt.close(fig)
        plt.close()
        plt.clf()
        del X, Y
        gc.collect()
    except:
        print(str(data[0])+'didnt work ....')

allhoms = []

for dronenumber in [1]:
    for vidnumber in [1]:
        homsdirpath = os.path.join(cut_data_dir,'DCIM-drone'+str(dronenumber),'drone'+str(dronenumber)+'vid'+str(vidnumber),homsdir)
        jpgsdirpath= os.path.join(cut_data_dir,'DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber))


        for file in os.listdir(homsdirpath):
            if file[-2:] == '.p':
                outfilepath = os.path.join(outputdir,file.replace('_hom.p','.jpg'))
                outhomfilepath = os.path.join(outputdir,file)
                framenumber = (file.split('frame')[-1]).split('_hom')[0]
                if int(framenumber) < 1770 or int(framenumber) > 1870:
                    continue
                else:
                    allhoms.append([os.path.join(homsdirpath,file),os.path.join(jpgsdirpath,file.replace('_hom.p','.jpg')),outhomfilepath,outfilepath])

#for fileframenumber in allhoms:
#	georectify(fileframenumber)



if __name__ == "__main__":
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(georectify, allhoms)