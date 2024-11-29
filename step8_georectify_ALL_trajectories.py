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
import ast

vidnumbers = [1,2,3,4,5]

matplotlib.use('Agg')

hom_diff_threshold = 0.4

homs_dir_to_use = 'correct_homs_18jan_all'
detections_dir_to_use = 'interpolated_detected'

outputdirpath = '/media/jean-pierre/PortableSSD/work_scratch'
final_detections_path = 'interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodlength_wooddiameter_woodvolume'

# Set the number of processes you want to use (adjust as needed)
num_processes = 8

curdir = os.getcwd()
divbyframes = 1

mainpath = os.path.join(curdir,'cut_data')

drone1frame_drone6frame_drone16frames = [
    [50,0,327],     #vid1
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

def find_closest_correct_hom_paths(hompath):
    framenumber_hompath = int((hompath.split('frame')[-1]).split('_hom')[0])
    
    #correct_hom_path = hompath.replace('/homs/'.'/correct_homs_18jan')
    #framenumber_correct_hom_path = int((correct_hom_path.split('frame')[-1]).split('_hom')[0])

    or_path_correct_homs = (hompath.split('/homs/')[0])+'/correct_homs_18jan'

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


def getlistofalldroneimages(mainpath,drone_vid_list,vidnumber,dronenumber):
    alldroneimages = []

    for folder in drone_vid_list:
        imagespath = os.path.join(mainpath,folder)
        for imfile in os.listdir(imagespath):
            if imfile[-4:] == '.jpg' and int(imfile[-9:-4]) % (fps/freq) == 0:
                hompath = os.path.join(mainpath,'DCIM-drone'+str(dronenumber),'drone'+str(dronenumber)+'vid'+str(vidnumber),homs_dir_to_use)
                homfile = imfile.replace('.jpg','_hom.p')

                #before_and_after = find_closest_correct_hom_paths(os.path.join(hompath,homfile))
                alldroneimages.append([os.path.join(imagespath,imfile),os.path.join(hompath,homfile)])

    alldroneimages.sort()
    #print(alldroneimages)

    return(alldroneimages)


def read_txt_file_old(file_path):
    '''Read a text file with U, V coordinates and ID

    Parameters
    ----------
    file_path : str
        Path to the text file

    Returns
    -------
    np.ndarray
        Nx1 matrix containing U coordinates (multiplied by 3840)
    np.ndarray
        Nx1 matrix containing V coordinates (multiplied by 2160)
    np.ndarray
        Nx1 matrix containing IDs
    '''
    data = np.loadtxt(file_path, usecols=(1, 2, 6))
    #print(data)
    U = data[:, 0] * 3840
    V = data[:, 1] * 2160
    IDs = data[:, 2]
    return U, V, IDs


def read_txt_file(file_path):
    '''Read a text file with U, V coordinates and ID

    Parameters
    ----------
    file_path : str
        Path to the text file

    Returns
    -------
    np.ndarray
        Nx1 matrix containing U coordinates (multiplied by 3840)
    np.ndarray
        Nx1 matrix containing V coordinates (multiplied by 2160)
    np.ndarray
        Nx1 matrix containing IDs
    '''
    data = np.loadtxt(file_path, usecols=(1, 2, 6), ndmin=2)
    U = data * [3840, 2160, 1]  # Multiply U by 3840, V by 2160, and leave IDs unchanged
    return U[:, 0], U[:, 1], U[:, 2]

def georectify_and_get_coordinates(U, V, H):
    '''Georectify U, V coordinates and get corresponding X, Y coordinates

    Parameters
    ----------
    U : np.ndarray
        Nx1 matrix containing U coordinates
    V : np.ndarray
        Nx1 matrix containing V coordinates
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        Nx1 matrix containing X coordinates
    np.ndarray
        Nx1 matrix containing Y coordinates
    '''
    X, Y = rectify_coordinates(U, V, H)
    return X, Y

def get_coordinates_and_wood_IDs(detection_txt_file,homography):
    H = homography

    # Replace 'your_data_file.txt' with the path to your actual data file
    #data_file_path = 'your_data_file.txt'

    # Read U, V coordinates and IDs from the text file
    U, V, IDs = read_txt_file(detection_txt_file)

    # Georectify and get corresponding X, Y coordinates
    X, Y = georectify_and_get_coordinates(U, V, H)

    # Combine X, Y coordinates with IDs
    result = np.column_stack((X, Y, IDs))

    return result

# Function to determine whether a point is left or right of the line
def point_position_line1(x, y):
    # Calculate the y-coordinate on the line for the given x-coordinate
    y_on_line = m_line1 * x + b_line1
    
    # Compare the y-coordinate of the point with the y-coordinate on the line
    if y < y_on_line:
        return "left"
    elif y > y_on_line:
        return "right"
    else:
        return "on the line"

# Function to determine whether a point is left or right of the line
def point_position_line2(x, y):
    # Calculate the y-coordinate on the line for the given x-coordinate
    y_on_line = m_line2 * x + b_line2
    
    # Compare the y-coordinate of the point with the y-coordinate on the line
    if y < y_on_line:
        return "left"
    elif y > y_on_line:
        return "right"
    else:
        return "on the line"

def read_detection_and_return_ID_plus_location(detections_path_drone1,detections_path_drone6,detections_path_drone16):
    #tl_tr_bl_br = []
    detections = []

    #read lines drone 1
    drone_number = 1
    try:
        with open(detections_path_drone1, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line_split=line.split(' ')
                ID = line_split[6]
                center = ast.literal_eval(line_split[8])
                tl = ast.literal_eval(line_split[9])
                tr = ast.literal_eval(line_split[10])
                bl = ast.literal_eval(line_split[11])
                br = ast.literal_eval(line_split[12])
                x_center, y_center = center
                if point_position_line1(x_center,y_center) == "right":
                    detections.append([ID,center,tl,tr,bl,br])
    except:
        no_detections = True

    #rint(detections_path_drone6)
    #read lines drone 6
    drone_number = 6
    try:
        with open(detections_path_drone6, 'r') as file:
            lines = file.readlines()
            #print(lines)
            for line in lines:
                line_split=line.split(' ')
                ID = line_split[6]
                center = ast.literal_eval(line_split[8])
                tl = ast.literal_eval(line_split[9])
                tr = ast.literal_eval(line_split[10])
                bl = ast.literal_eval(line_split[11])
                br = ast.literal_eval(line_split[12])
                x_center, y_center = center
                #print('leftright_+_______')
                #print(point_position_line1(x_center,y_center))
                #print(point_position_line2(x_center,y_center))
                #print(' ')
                if point_position_line1(x_center,y_center) == "left":
                    if point_position_line2(x_center,y_center) == "right":
                        detections.append([ID,center,tl,tr,bl,br])
    except:
        no_detections = True

    #read lines drone 16
    drone_number = 16
    try:
        with open(detections_path_drone16, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line_split=line.split(' ')
                ID = line_split[6]
                center = ast.literal_eval(line_split[8])
                tl = ast.literal_eval(line_split[9])
                tr = ast.literal_eval(line_split[10])
                bl = ast.literal_eval(line_split[11])
                br = ast.literal_eval(line_split[12])
                x_center, y_center = center
                if point_position_line2(x_center,y_center) == "left":
                    detections.append([ID,center,tl,tr,bl,br])
    except:
        no_detections = True

    if len(detections) == 0:
        detections = None

    return(detections)





data_to_plot = []

ALL_remembered_lines = []

for vidnumber in vidnumbers:
    stagered_drone_ims = []

    outputdir = os.path.join(outputdirpath,'vid'+str(vidnumber))
    os.makedirs(outputdir, exist_ok = True)

    #now load the stagering ( [49,0,328] )
    drone1frame_drone6frame_drone16frame = drone1frame_drone6frame_drone16frames[vidnumber-1]
    #print(drone1frame_drone6frame_drone16frame)

    vid_drone1 = ['DCIM-drone1_drone1vid'+str(vidnumber)]
    vid_drone6 = ['DCIM-drone6_drone6vid'+str(vidnumber)]
    vid_drone16 = ['DCIM-drone16_drone16vid'+str(vidnumber)]

    alldrone_1_images_and_homs = getlistofalldroneimages(mainpath,vid_drone1,vidnumber,1)
    alldrone_6_images_and_homs =  getlistofalldroneimages(mainpath,vid_drone6,vidnumber,6)
    alldrone_16_images_and_homs =  getlistofalldroneimages(mainpath,vid_drone16,vidnumber,16)

    framenumber = 0
    for i in range(0,200000):
        framenumber = framenumber + 1

        if i-drone1frame_drone6frame_drone16frame[0] < 0:
            drone1im = None
            drone1hom = None
        else:
            try:
                drone1im = alldrone_1_images_and_homs[i-drone1frame_drone6frame_drone16frame[0]][0]
                drone1hom = alldrone_1_images_and_homs[i-drone1frame_drone6frame_drone16frame[0]][1]
                #print(drone1hom)
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
                #print(drone6hom)
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
                #print(drone16hom)
            except:
                drone16im = None
                drone16hom = None
        #print(drone16im)
        if drone1im == None and drone6im == None and drone16im == None:
            break

        if drone1im != None:
            drone1framenum = (drone1im.split('_frame')[-1]).split('.jpg')[0]
        else:
            drone1framenum = 'None'
        if drone6im != None:
            drone6framenum = (drone6im.split('_frame')[-1]).split('.jpg')[0]
        else:
            drone6framenum = 'None'
        if drone16im != None:
            drone16framenum = (drone16im.split('_frame')[-1]).split('.jpg')[0]
        else:
            drone16framenum = 'None'

        output_file_path = os.path.join(outputdir,'frame'+str(framenumber).zfill(5)+'_drone1frame'+drone1framenum+'_drone6frame'+drone6framenum+'_drone16frame'+drone16framenum+'.jpg')

        if drone1hom != None:
            drone1detections = (drone1hom.replace(homs_dir_to_use,detections_dir_to_use)).replace('_hom.p','.txt')
        else:
            drone1detections = None
        if drone6hom != None:
            drone6detections = (drone6hom.replace(homs_dir_to_use,detections_dir_to_use)).replace('_hom.p','.txt')
        else:
            drone6detections = None
        if drone16hom != None:
            drone16detections = (drone16hom.replace(homs_dir_to_use,detections_dir_to_use)).replace('_hom.p','.txt')
        else:
            drone16detections = None

        #homs_dir_to_use = 'correct_homs_18jan_all'
        #detections_dir_to_use = 'allparts'

        stagered_drone_ims.append([[drone1im,drone1hom,drone1detections],[drone6im,drone6hom,drone6detections],[drone16im,drone16hom,drone16detections],output_file_path])



        #stagered_drone_ims.append([[drone1_image_path,drone1_homography_path,drone1_detections_path],[drone6_image_path,drone6_homography_path,drone6_detections_path],[drone16_image_path,drone16_homography_path,drone16_detections_path],output_file_path])

        #stagered_drone_ims.append([[drone1im,drone6im,drone16im],[drone1hom,drone6hom,drone16hom],os.path.join(outputdir,'frame'+str(framenumber).zfill(5)+'_drone1frame'+drone1framenum+'_drone6frame'+drone6framenum+'_drone16frame'+drone16framenum+'.jpg'),[vidnumber,drone1framenum,drone6framenum,drone16framenum,outputdir]])


    #for vidnumber in [1]:

    drone_id_coordinates = {}

    
    #for data in stagered_drone_ims:
    #def georectify_three_images(data,remembered_lines):
    remembered_lines_last = []
    remembered_lines = []
    points = []
    for data in stagered_drone_ims:
        
        output_image_path = data[3]

        #if os.path.exists(output_image_path) != True:
        drone1_image_path = data[0][0]
        drone1_homography_path = data[0][1]
        drone1_detections_path = data[0][2]
        if drone1_homography_path != None:
            drone1_final_detections_path = (drone1_homography_path.replace('correct_homs_18jan_all',final_detections_path)).replace('_hom.p','.txt')
        else:
            drone1_final_detections_path = None

        drone6_image_path = data[1][0]
        #print(drone6_image_path)
        drone6_homography_path = data[1][1]
        #print(drone6_homography_path)
        drone6_detections_path = data[1][2]
        if drone6_homography_path != None:
            drone6_final_detections_path = (drone6_homography_path.replace('correct_homs_18jan_all',final_detections_path)).replace('_hom.p','.txt')
        else:
            drone6_final_detections_path = None
        
        drone16_image_path = data[2][0]
        drone16_homography_path = data[2][1]
        drone16_detections_path = data[2][2]
        if drone16_homography_path != None:
            drone16_final_detections_path = (drone16_homography_path.replace('correct_homs_18jan_all',final_detections_path)).replace('_hom.p','.txt')
        else:
            drone16_final_detections_path = None
        
        detections = read_detection_and_return_ID_plus_location(drone1_final_detections_path,drone6_final_detections_path,drone16_final_detections_path)


        points = []
        remembered_lines_last = []
        if detections != None:
            for detection in detections:
                old_coordinates = None

                if detection[0] in drone_id_coordinates:
                    old_coordinates = drone_id_coordinates[detection[0]]
                
                tl = detection[2]
                tr = detection[3]
                bl = detection[4]
                br = detection[5]
                remembered_lines_last.append([[tl,tr],[tr,br],[br,bl],[bl,tl]])

                #x1_top,y1_top = 

                new_coordinates = detection[1]
                x_new, y_new = new_coordinates

                points.append([x_new,y_new])

                if old_coordinates != None:
                    x1,y1 = old_coordinates
                    x2,y2 = new_coordinates
                    remembered_lines.append([old_coordinates,new_coordinates])
                    

                #drone_number = detection[1]
                id_number = detection[0]
                coordinates = detection[1]

                # Check if the drone number already exists in the dictionary
                if id_number in drone_id_coordinates:
                    # If ID exists, update the coordinates
                    drone_id_coordinates[id_number] = coordinates
                else:
                    # If ID doesn't exist, add it to the dictionary for that drone
                    drone_id_coordinates[id_number] = coordinates
        
        #print(remembered_lines)
        data_to_plot.append([data,remembered_lines.copy(),remembered_lines_last])

        #print(len(data_to_plot[0][1]))
        #print('at end: ',len(data_to_plot[0][1]))
    #print('at far end: ',len(data_to_plot[0][1]))
    #print(data_to_plot[0][1])
    ALL_remembered_lines.append(remembered_lines.copy())



#print(data_to_plot)





#for data in stagered_drone_ims:


#print(data_to_plot[0])
data = data_to_plot[0]
remembered_lines_georec_ = ALL_remembered_lines
#print(len(remembered_lines_georec))
#remembered_lines_last = data_to_plot[2]


drone1_image_path = '/media/jean-pierre/PortableSSD/DCIM-drone1_drone1vid4_frame06336.jpg'
drone1_homography_path = '/media/jean-pierre/PortableSSD/DCIM-drone1_drone1vid4_frame06336_hom.p'

drone6_image_path = '/media/jean-pierre/PortableSSD/DCIM-drone6_drone6vid4_frame06312.jpg'
drone6_homography_path = '/media/jean-pierre/PortableSSD/DCIM-drone6_drone6vid4_frame06312_hom.p'

drone16_image_path = '/media/jean-pierre/PortableSSD/DCIM-drone16_drone16vid4_frame06138.jpg'
drone16_homography_path = '/media/jean-pierre/PortableSSD/DCIM-drone16_drone16vid4_frame06138_hom.p'

output_image_path = '/media/jean-pierre/PortableSSD/georec_all_lines.jpg'
print(output_image_path)






figsize = (70, 30)
#figsize = (20, 10)
cmap = 'Greys'
color = True
n_alpha = 0

fig, ax = plt.subplots(figsize=figsize)



try:
    #DRONE 6
    if drone6_image_path == None:
        pointlessparameter = 1
        #print('no Drone 6 image')
    else:
        drone6_image = cv2.imread(drone6_image_path)

        with open(drone6_homography_path, 'rb') as pickle_f:
            drone6_homography = pickle.load(pickle_f)

        
        X, Y = rectify_image(drone6_image, drone6_homography)

        bgrimg = drone6_image
        img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)



        




        if color:
            rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
            im.set_array(None)
            im.set_edgecolor('none')
            im.set_facecolor(rgba)
            im.set_alpha(opacity)
    
except:
    print('Drone 6 DIDNT WORK')

try:
    #DRONE 1
    if drone1_image_path == None:
        pointlessparameter = 1
        #print('no Drone 1 image')
    else:
        drone1_image = cv2.imread(drone1_image_path)

        with open(drone1_homography_path, 'rb') as pickle_f:
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
    
except:
    print('Drone 1 DIDNT WORK')





try:
    #DRONE 16
    if drone16_image_path == None:
        pointlessparameter = 1
        #print('no Drone 16 image')
    else:
        drone16_image = cv2.imread(drone16_image_path)

        with open(drone16_homography_path, 'rb') as pickle_f:
            drone16_homography = pickle.load(pickle_f)

        X, Y = rectify_image(drone16_image, drone16_homography)

        bgrimg = drone16_image
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
    del X, Y
    
except:
    print('Drone 16 DIDNT WORK')

#print('length',len(remembered_lines_georec))


for i in range(5):
    remembered_lines_georec = remembered_lines_georec_[i]
    for remembered_line in remembered_lines_georec:
        x1,y1=remembered_line[0]
        x2,y2=remembered_line[1]
        plt.plot([x1, x2], [y1, y2], color='white', linestyle='-', linewidth=2, alpha=0.4)


Section6 = ((2804162.1299-Xbase,1175290.8655-Ybase),(2804181.03779-Xbase,1175253.16418-Ybase))
Section5 = ((2804132.9050-Xbase,1175280.4683-Ybase),(2804142.0906-Xbase,1175249.4849-Ybase))
Section4 = ((2804104.0364-Xbase,1175274.5377-Ybase),(2804106.5156-Xbase,1175248.0643-Ybase))
Section3 = ((2804056.8109-Xbase,1175273.5760-Ybase),(2804070.4716-Xbase,1175236.7660-Ybase))
Section2 = ((2804033.1186-Xbase,1175264.8710-Ybase),(2804059.1070-Xbase,1175228.7094-Ybase))
Section1 = ((2804005.0938-Xbase,1175250.3824-Ybase),(2804041.5581-Xbase,1175221.2789-Ybase))

Section1 = ((2804162.1299-Xbase,1175290.8655-Ybase),(2804181.03779-Xbase,1175253.16418-Ybase))
Section2 = ((2804132.9050-Xbase,1175280.4683-Ybase),(2804142.0906-Xbase,1175249.4849-Ybase))
Section3 = ((2804104.0364-Xbase,1175274.5377-Ybase),(2804106.5156-Xbase,1175248.0643-Ybase))
Section4 = ((2804056.8109-Xbase,1175273.5760-Ybase),(2804070.4716-Xbase,1175236.7660-Ybase))
Section5 = ((2804033.1186-Xbase,1175264.8710-Ybase),(2804059.1070-Xbase,1175228.7094-Ybase))
Section6 = ((2804005.0938-Xbase,1175250.3824-Ybase),(2804041.5581-Xbase,1175221.2789-Ybase))

sections_and_numbers = [
    [Section1,1],
    [Section2,2],
    [Section3,3],
    [Section4,4],
    [Section5,5],
    [Section6,6],  
    ]

for section,number in sections_and_numbers:    
    plt.plot([section[0][0], section[1][0]], [section[0][1], section[1][1]], linestyle='--', linewidth=10, color='lightsteelblue')
    plt.text(section[0][0]-4, section[0][1]+1, 'Section '+str(number), fontsize = 70, color='white')
    #plt.text(section[0][0]-1, section[0][1]+1, str(number), fontsize = 70, color='white')
    #plt.text(section[1][0]-1, section[1][1]-3, str(number)+"'", fontsize = 70, color='white')

plt.arrow(200,25,0,5,width=0.7,color='black')
plt.text(200, 34, 'N', fontsize = 70, color='black', ha='center')




ax.set_aspect('equal')

#plt.text(20, 20, str(data[2]), size='xx-large')
plt.ylim(1175215-Ybase, 1175295-Ybase)
plt.xlim(2804000-Xbase, 2804190-Xbase)
#plt.xlabel('y (m)', fontsize=40)
#plt.xlabel('x (m)', fontsize=40)

# Define the length and width of the scale indicator
length = 40
width = 1
start_of_scale = 155

# Calculate the number of segments
num_segments = 4

# Calculate the width of each segment
segment_width = length / num_segments

# Get the current axes
ax = plt.gca()

# Plot the scale indicator and add annotations
for i in range(num_segments):
    if i % 2 == 0:
        color = 'black'  # Black color for even segments
    else:
        color = 'white'  # White color for odd segments
    
    # Calculate the coordinates of the rectangle
    x = start_of_scale + i * segment_width
    y = 25 - width / 2
    
    # Draw the rectangle with a black edge
    ax.fill([x, x + segment_width, x + segment_width, x], [y, y, y + width, y + width], color=color, edgecolor='black', linewidth=0.5, transform=ax.transData)
    
    # Add text annotation for the length of each segment
    segment_length = segment_width * 100  # Convert to centimeters
    segment_length_meters = segment_length / 100
    segment_length_meters_seg = int(segment_length_meters * i)
    
    ax.text(x, y - 1, f'{segment_length_meters_seg}', ha='center', va='center', fontsize=30)

    if i == num_segments-1:
        segment_length_meters_seg = int(segment_length_meters * (i+1))
        ax.text(x + segment_width, y - 1, f'{segment_length_meters_seg}', ha='center', va='center', fontsize=30)


ax.text(start_of_scale + (num_segments*segment_width)/2, 26+width/2, 'Meters', ha='center', va='center', fontsize=30)

# Set the aspect ratio to equal for a square scale indicator
ax.set_aspect('equal', adjustable='box')

# Hide ticks and adjust label size for both x and y axes
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelsize=14)

# Hide the axis
plt.axis('off')

plt.arrow(178,42,-5,-0.75,width=0.7,color='black')
plt.text(174.5, 43, 'Flow Direction', fontsize = 40, color='black', ha='center')

#plt.legend()
plt.tight_layout()
#print(data[2])
#plt.savefig(data[2])
plt.savefig(output_image_path)

fig.clf()
plt.close(fig)
plt.close()
plt.clf()
        
gc.collect()