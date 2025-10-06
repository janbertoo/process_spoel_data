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

curdir = os.getcwd()
divbyframes = 1

#vidnumber_name = '1_1'
vidnumber = 1
mainpath = os.path.join(curdir,'cut_data')

drone1frame_drone6frame_drone16frame = [49,0,328]   #vid1
#drone1frame_drone6frame_drone16frame = [60,0,211]   #vid2
#drone1frame_drone6frame_drone16frame = [30,0,115]   #vid3
#drone1frame_drone6frame_drone16frame = [24,0,22]    #vid4
#drone1frame_drone6frame_drone16frame = [0,91,1]    #vid5

drone1frame_drone6frame_drone16frames = [
    [49,0,328],     #vid1
    [60,0,211],     #vid2
    [30,0,115],     #vid3
    [24,0,22],      #vid4
    [0,91,1],        #vid5
]

outputdir = os.path.join(curdir,'vid'+(str(vidnumber)))
#outputdir = os.path.join('/home/jean-pierre/Desktop/8dec','vid'+str(vidnumber))
fps = 24
freq = 4
freq_old = 2

x=3840
y=2160

U, V = np.meshgrid(range(x),range(y))

#define base coordinates
Xbase = 2803987
Ybase = 1175193

divby = 1
altcor = 0
slopecor = 1

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


m_line1 = (y1_line1-y2_line1)/(x1_line1-x2_line1)                           #slope
b_line1 = (x1_line1*y2_line1 - x2_line1*y1_line1)/(x1_line1-x2_line1)       #y-intercept

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

def create_a_virtual_gcp_from_2_gcps(GCP1,GCP2):
    virGCP_uv = ( (GCP1[0][0]+GCP2[0][0]) / 2 , (GCP1[0][1]+GCP2[0][1]) / 2)
    virGCP_rw = ( (GCP1[1][0]+GCP2[1][0]) / 2 , (GCP1[1][1]+GCP2[1][1]) / 2 , (GCP1[1][2]+GCP2[1][2]) / 2 )

    return(virGCP_uv,virGCP_rw)

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

def getlistofalldroneimages(mainpath,drone_vid_list):
    alldroneimages = []

    for folder in drone_vid_list:
        imagespath = os.path.join(mainpath,folder)
        for imfile in os.listdir(imagespath):
            if imfile[-4:] == '.jpg' and int(imfile[-9:-4]) % (fps/freq) == 0:
                alldroneimages.append(os.path.join(imagespath,imfile))

    alldroneimages.sort()

    return(alldroneimages)

def get_list_of_txts(path):
    txtlist = []
    for file in os.listdir(path):
        if file[-4:] == '.txt':
            txtlist.append(file)
    txtlist.sort()

    return(txtlist)

def get_list_of_ps(path):
    plist = []
    for file in os.listdir(path):
        if file[-2:] == '.p':
            plist.append(file)
    plist.sort()

    return(plist)

def get_pixel_and_rw_coords(txt_file):
    gcps_pixel_and_coors = []

    with open(txt_file, "r") as txt_file:
        lines_txt_file = txt_file.readlines()

        for line in lines_txt_file:
            splitted = line.split(', ')
            pixelcoor = float(splitted[0].split(' ')[0]),float(splitted[0].split(' ')[1])
            rwcoor = float(splitted[1].split(' ')[0])-Xbase,float(splitted[1].split(' ')[1])-Ybase,   ( float(splitted[1].split(' ')[2]) - waterlevel -   get_corrected_zcoor( float(splitted[1].split(' ')[0]) , float(splitted[1].split(' ')[1]) ) )     /divby
            gcps_pixel_and_coors.append([pixelcoor,rwcoor])

    return(gcps_pixel_and_coors)

#dronenumber = 1
#vidnumber = 1

#for dronenumber in [1,6,16]:
    #for vidnumber in [1,2,3,4,5]:
    

def process_viddrone(dronenumbervidnumber):
    dronenumber = dronenumbervidnumber[0]
    vidnumber = dronenumbervidnumber[1]
    print('drone'+str(dronenumber)+'_vid'+str(vidnumber))

    labels_with_gcps_path = os.path.join(curdir,'cut_data/DCIM-drone'+str(dronenumber)+'/drone'+str(dronenumber)+'vid'+str(vidnumber)+'/labels_with_gcps')
    #print(labels_with_gcps_path)
    homography_path = labels_with_gcps_path.replace('labels_with_gcps','homs_remove1_19jan')
    os.makedirs(homography_path, exist_ok=True)

    correct_homography_path = labels_with_gcps_path.replace('labels_with_gcps','correct_homs')

    correct_homography_list = get_list_of_ps(correct_homography_path)
    labels_with_gcps_txtlist = get_list_of_txts(labels_with_gcps_path)

    all_info = []

    for labels_with_gcps_txt in labels_with_gcps_txtlist:
        framenumber = int((labels_with_gcps_txt.split('_frame')[-1]).split('.txt')[0])
        full_path_labels_with_gcps_txt = os.path.join(labels_with_gcps_path,labels_with_gcps_txt)
        full_path_homography_p_file = os.path.join(homography_path,labels_with_gcps_txt.replace('.txt','_hom.p'))

        closest = 1000
        #find closest correct homography to currect image
        for correct_homography in os.listdir(correct_homography_path):
            #print(correct_homography)
            framenumber_cor_hom = int((correct_homography.split('_frame')[-1]).split('_hom.p')[0])
            diff = np.abs( framenumber - framenumber_cor_hom )
            
            if diff < closest:
                full_path_correct_homography_p_file = os.path.join(correct_homography_path,correct_homography)
                closest = diff

        all_info.append([vidnumber,dronenumber,framenumber,full_path_labels_with_gcps_txt,full_path_homography_p_file,full_path_correct_homography_p_file])


    max_diff = 0
    all_homographies = []
    all_optimized_homographies = []
    all_framenumbers = []

    for entry in all_info:
        virtualGCPS_combos = []
        diff_and_homs = []
        drones_used = []
        removedGCP = ['none']
        #print(entry)
        #load closest correct homography
        close_correct_homography_pickle_file = entry[5]
        with open(close_correct_homography_pickle_file, 'rb') as pickle_f:
            close_correct_homography = pickle.load(pickle_f)

        #determine drone internal camera parameters
        if entry[1] == 1:
            camera_matrix = camera_matrix_drone1
            distortion_coefficients = distortion_coefficients_drone1
        if entry[1] == 6:
            camera_matrix = camera_matrix_drone6
            distortion_coefficients = distortion_coefficients_drone6
        if entry[1] == 16:
            camera_matrix = camera_matrix_drone16
            distortion_coefficients = distortion_coefficients_drone16
        
        #get coodinates from txt file
        txt = entry[3]
        coorstxt = get_pixel_and_rw_coords(txt)
        #print(coorstxt)

        #define rw coordinates
        xyzs = []
        for i in range(len(coorstxt)):
            xyzs.append(coorstxt[i][1])
            drones_used.append(i)
        XYZ = np.array(xyzs, dtype=np.float32)
        
        #define pixel coordinates
        uvs = []
        for i in range(len(coorstxt)):
            uvs.append(coorstxt[i][0])
        UV = np.array(uvs, dtype=np.float32)

        #culculate homography
        homography = find_homography(UV, XYZ, camera_matrix, distortion_coefficients)

        #calculate difference homographies
        totaldiff = 0
        for q in range(len(homography)):
            for w in range(len(homography[0])):
                totaldiff = totaldiff + np.abs( homography[q][w] - close_correct_homography[q][w] )
        #print(totaldiff)

        count = 0
        diff_and_homs.append([totaldiff,count,drones_used,virtualGCPS_combos,homography,removedGCP])
        count = count + 1

        #add 1 and 2 virtual GCPS
        #amount of tries with a virtualGCP
        tries_vir_GCPS = 3

        for i in range(tries_vir_GCPS):
            removedGCP = ['none']
            virtualGCPS_combos = []
            xyzs_with_vir = xyzs
            uvs_with_vir = uvs

            #create a random GCPS
            rand1 = drones_used[np.random.randint(0,len(drones_used))]
            rand2 = drones_used[np.random.randint(0,len(drones_used))]
            while rand2 == rand1:
                rand2 = drones_used[np.random.randint(0,len(drones_used))]

            virtualGCP = create_a_virtual_gcp_from_2_gcps(coorstxt[rand1],coorstxt[rand2])

            virtualGCPS_combos = [rand1,rand2]

            xyzs_with_vir.append(virtualGCP[1])
            uvs_with_vir.append(virtualGCP[0])

            XYZ = np.array(xyzs_with_vir, dtype=np.float32)
            UV = np.array(uvs_with_vir, dtype=np.float32)

            #culculate homography
            try:
                op_homography = find_homography(UV, XYZ, camera_matrix, distortion_coefficients)
                #print(op_homography)
            except:
                op_homography = np.array([[10,10,10],[10,10,10],[10,10,10]])
            
            op_totaldiff = 0
            for q in range(len(op_homography)):
                for w in range(len(op_homography[0])):
                    op_totaldiff = op_totaldiff + np.abs( op_homography[q][w] - close_correct_homography[q][w] )
            
            diff_and_homs.append([op_totaldiff,count,drones_used,virtualGCPS_combos,op_homography,removedGCP])
            count = count + 1


        
        



        ############OPTIMIZE HOMOGRAPHY remove 1 gcp
        for i in range(len(coorstxt)):
            print(coorstxt[i])
            print(coorstxt[i][0])
            removedGCP = str(coorstxt[i][1]).replace(' ','_')
            drones_used = []
            #define rw coordinates
            xyzs = []
            virtualGCPS_combos = []
            for j in range(len(coorstxt)):
                if j != i:
                    xyzs.append(coorstxt[j][1])
                    drones_used.append(j)
            XYZ = np.array(xyzs, dtype=np.float32)
            
            #define pixel coordinates
            uvs = []
            for j in range(len(coorstxt)):
                if j != i:
                    uvs.append(coorstxt[j][0])
            UV = np.array(uvs, dtype=np.float32)

            #culculate homography
            try:
                op_homography = find_homography(UV, XYZ, camera_matrix, distortion_coefficients)
            except:
                op_homography = np.array([[10,10,10],[10,10,10],[10,10,10]])
            
            op_totaldiff = 0
            for q in range(len(op_homography)):
                for w in range(len(op_homography[0])):
                    op_totaldiff = op_totaldiff + np.abs( op_homography[q][w] - close_correct_homography[q][w] )

            
            diff_and_homs.append([op_totaldiff,count,drones_used,virtualGCPS_combos,op_homography,removedGCP])
            count = count + 1

            



            

        


        diff_and_homs.sort()

        all_homographies.append(totaldiff)
        all_optimized_homographies.append(diff_and_homs[0][0])
        all_framenumbers.append(entry[2])
        #print(entry[4])
        #print(diff_and_homs[0][4])
        for data in diff_and_homs:
            pickle_file = open(entry[4].replace('_hom.p',str(data[5])+'_hom.p'), 'wb')
            pickle.dump(data[4], pickle_file)


    plt.plot(all_framenumbers,all_homographies)
    ax = plt.gca()
    ax.set_ylim([0, 0.5])
    plt.savefig('homography_analysis/figure_drone'+str(dronenumber)+'_vid'+str(vidnumber)+'_Aoriginal.svg')
    plt.plot(all_framenumbers,all_optimized_homographies)
    plt.savefig('homography_analysis/figure_drone'+str(dronenumber)+'_vid'+str(vidnumber)+'_optimized.svg')
    plt.clf()








if __name__ == "__main__":
    # Set the number of processes you want to use (adjust as needed)
    num_processes = 2

    files = []
    for dronenumber in [1,6,16]:
        for vidnumber in [1,2,3,4,5]:
            files.append([dronenumber,vidnumber])

    files = []
    for dronenumber in [16]:
        for vidnumber in [1,2,3,4,5]:
            files.append([dronenumber,vidnumber])

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the processing function to the list of SVG files
        pool.map(process_viddrone, files)