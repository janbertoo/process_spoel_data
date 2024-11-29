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
outputdir = '/home/jean-pierre/Desktop/13feb/surface_vids'

cut_data_dir = os.path.join(curdir,'cut_data')
cut_data_dir_homs = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data'

homsdir = 'correct_homs_18jan'
homsdir = 'homs'

#define base coordinates
Xbase = 2803987
Ybase = 1175193

num_processes = 8


def find_closest_correct_hom_paths(hompath):
    print(hompath)
    print('')
    framenumber_hompath = int((hompath.split('frame')[-1]).split('_hom')[0])
    print(framenumber_hompath)

    or_path_correct_homs = (hompath.split('/homs/')[0])+'/correct_homs_18jan'
    print(or_path_correct_homs)

    before_and_after_list_smaller = []
    before_and_after_list_larger = []

    before_and_after = None
    
    for homfile in os.listdir(or_path_correct_homs):
        correct_hom_path = os.path.join(or_path_correct_homs,homfile)
        framenumber_correct_hom_path = int((homfile.split('frame')[-1]).split('_hom')[0])
        
        if framenumber_correct_hom_path == framenumber_hompath:
            before_and_after = [correct_hom_path,correct_hom_path,0,1,2]
        if framenumber_correct_hom_path < framenumber_hompath:
            before_and_after_list_smaller.append([framenumber_hompath-framenumber_correct_hom_path,framenumber_correct_hom_path,framenumber_correct_hom_path,correct_hom_path])
        if framenumber_correct_hom_path > framenumber_hompath:
            before_and_after_list_larger.append([framenumber_correct_hom_path-framenumber_hompath,framenumber_correct_hom_path,framenumber_correct_hom_path,correct_hom_path])

    before_and_after_list_smaller.sort()
    before_and_after_list_larger.sort()

    if before_and_after == None:
        if len(before_and_after_list_smaller) == 0:
            before_smaller = before_and_after_list_larger[0][3]
            frameno_smaller = 0
        else:
            before_smaller = before_and_after_list_smaller[0][3]
            frameno_smaller = before_and_after_list_smaller[0][2]

        if len(before_and_after_list_larger) == 0:
            before_larger = before_and_after_list_smaller[0][3]
            frameno_larger = 99999999999999999
        else:
            before_larger = before_and_after_list_larger[0][3]
            frameno_larger = before_and_after_list_larger[0][2]

        before_and_after = [before_smaller,before_larger,frameno_smaller,framenumber_hompath,frameno_larger]

    return(before_and_after)

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
        hom_1 = data[0][0]
        hom_2 = data[0][1]

        hom_small_frame = data[0][2]
        hom_cur_frame = data[0][3]
        hom_large_frame = data[0][4]



        jpgfile = data[1]
        outhomfilepath = data[2]
        outfilepath = data[3]

        figsize = (60, 40)
        cmap = 'Greys'
        color = True
        n_alpha = 0

        fig, ax = plt.subplots(figsize=figsize)

        image = cv2.imread(jpgfile)

        #with open(homfile, 'rb') as pickle_f1:
        #	homography = pickle.load(pickle_f1)

        with open(hom_1, 'rb') as pickle_f1:
            homography_1 = pickle.load(pickle_f1)
        with open(hom_2, 'rb') as pickle_f2:
            homography_2 = pickle.load(pickle_f2)

        homography = np.array([[float(),float(),float()],[float(),float(),float()],[float(),float(),float()]])
        for o in range(3):
            for p in range(3):
                largeportion = ( ( hom_cur_frame - hom_small_frame ) / ( hom_large_frame - hom_small_frame ) )
                smallportion = ( ( hom_large_frame - hom_cur_frame ) / ( hom_large_frame - hom_small_frame ) )
                homography[o,p] = smallportion * homography_1[o,p] + largeportion * homography_2[o,p]

        X, Y = rectify_image(image, homography)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

        rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
        im.set_array(None)
        im.set_edgecolor('none')
        im.set_facecolor(rgba)

        ax.set_aspect('equal')

        plt.ylim(1175215-Ybase, 1175270-Ybase)
        plt.xlim(2803995-Xbase, 2804075-Xbase)
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

dronenumber = 16
vidnumbers = [5]

for vidnumber in vidnumbers:
    homsdirpath = os.path.join(cut_data_dir_homs,'DCIM-drone'+str(dronenumber),'drone'+str(dronenumber)+'vid'+str(vidnumber),homsdir)
    jpgsdirpath= os.path.join(cut_data_dir,'DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber))
    print(homsdirpath)
    print(jpgsdirpath)


    for file in os.listdir(jpgsdirpath):
        if file[-4:] == '.jpg':
            outfilepath = os.path.join(outputdir+'_d'+str(dronenumber)+'v'+str(vidnumber),file)#.replace('_hom.p','.jpg'))
            if os.path.exists(outputdir+'_d'+str(dronenumber)+'v'+str(vidnumber)) == False:
                os.mkdir(outputdir+'_d'+str(dronenumber)+'v'+str(vidnumber))
            print(outfilepath)
            outhomfilepath = os.path.join(outputdir+'_d'+str(dronenumber)+'v'+str(vidnumber),file.replace('.jpg','_hom.p'))
            print(outhomfilepath)
            framenumber = (file.split('frame')[-1]).split('.jpg')[0]
            print(framenumber)
            #if int(framenumber) < 1770 or int(framenumber) > 1870:
            #    continue
            #else:
            #print(os.path.join(homsdirpath,file))
            before_and_after = find_closest_correct_hom_paths(os.path.join(homsdirpath,file.replace('.jpg','_hom.p')))
            allhoms.append([before_and_after,os.path.join(jpgsdirpath,file.replace('_hom.p','.jpg')),outhomfilepath,outfilepath])



if __name__ == "__main__":
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(georectify, allhoms)



for vidnumber in vidnumbers:
    os.system("ffmpeg -framerate 24 -pattern_type glob -i '/home/jean-pierre/Desktop/13feb/surface_vids_d"+str(dronenumber)+"v"+str(vidnumber)+"/*.jpg' /home/jean-pierre/Desktop/13feb/surface_vids_d"+str(dronenumber)+"v"+str(vidnumber)+"/drone"+str(dronenumber)+"vid"+str(vidnumber)+".mp4")