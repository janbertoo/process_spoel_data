import os
import pickle
import math
import numpy as np
import cv2

curdir = os.getcwd()
#mainpath = os.path.join(curdir,'cut_data')

x_width = 3840
y_height = 2160

vidnumber = 4

vid_path = os.path.join(curdir,'vid'+str(vidnumber))

output_path = os.path.join(curdir,'vid'+str(vidnumber),'inferred_rw_coordinates')

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

def write_rw_coordinates_from_inferred(txt,output_file,homography_pickle_file):    
    #load homography from pickled file
    with open(homography_pickle_file, 'rb') as pickle_f:
        hom = pickle.load(pickle_f)

    U_list = []
    V_list = []

    #create variable for top left TL and BR bottom right of the bounding box
    TL_U_list = []
    TL_V_list = []
    BR_U_list = []
    BR_V_list = []

    conf_list = []

    with open(txt) as txt_file:
        lines = txt_file.readlines()

    for line in lines:
        line_split = line.split(' ')

        if line_split[0] == '0':   # 0 means wood
            U_list.append( x_width * float(line_split[1]) )
            V_list.append( y_height * float(line_split[2]) )

            TL_U_list.append( int( math.floor( ( float(line_split[1]) - ( float(line_split[3]) / 2 ) ) * x_width ) ) )
            TL_V_list.append( int( math.floor( ( float(line_split[2]) - ( float(line_split[4]) / 2 ) ) * y_height ) ) )
            
            BR_U_list.append( int( math.floor( ( float(line_split[1]) + ( float(line_split[3]) / 2 ) ) * x_width ) ) )
            BR_V_list.append( int( math.floor( ( float(line_split[2]) + ( float(line_split[4]) / 2 ) ) * y_height ) ) )

            conf_list.append(line_split[5])

    U = np.asarray(U_list)
    V = np.asarray(V_list)

    TL_U = np.asarray(TL_U_list)
    TL_V = np.asarray(TL_V_list)
    BR_U = np.asarray(BR_U_list)
    BR_V = np.asarray(BR_V_list)

    X, Y = rectify_coordinates(U, V, hom)

    TL_X, TL_Y = rectify_coordinates(TL_U, TL_V, hom)
    BR_X, BR_Y = rectify_coordinates(BR_U, BR_V, hom)

    with open(output_file, 'w') as output_f:
        for i in range(len(X)):
            output_f.write('0 '+str(X[i])+' '+str(Y[i])+' '+str(TL_X[i])+' '+str(TL_Y[i])+' '+str(BR_X[i])+' '+str(BR_Y[i])+' '+conf_list[i])

#create folder if not exists
if os.path.exists(output_path) == False:
    os.mkdir(output_path)

for file in os.listdir(vid_path):
    if file[-6:] == '_hom.p':
        split_file = (file.split('drone')[-1]).split('vid')
        drone_number = split_file[0]

        txt_file_name = ('DCIM'+(file.split('_DCIM')[-1])).replace('_hom.p','.txt')
        
        txt_file_path = os.path.join(curdir,'cut_data/DCIM-drone'+drone_number+'/drone'+drone_number+'vid'+str(vidnumber)+'/inferred/',txt_file_name)
        hom_file_path = os.path.join(vid_path,file)
        out_file_path = (os.path.join(output_path,file)).replace('_hom.p','.txt')

        try:
            write_rw_coordinates_from_inferred(txt_file_path,out_file_path,hom_file_path)
        except:
            print("DIDN'T WORK "+txt_file_name)
