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
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

drone1impath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone1_drone1vid2_frame01434.jpg'
drone1hompath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone1_drone1vid2_frame01434_hom.p'

drone6impath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone6_drone6vid2_frame01794.jpg'
drone6hompath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone6_drone6vid2_frame01794_hom.p'

drone16impath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone16_drone16vid2_frame00528.jpg'
drone16hompath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/DCIM-drone16_drone16vid2_frame00528_hom.p'

outputimpath = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities.jpg'

plt.rcParams.update({'font.size': 40})
threshold_value = 6

matplotlib.use('Agg')

hom_diff_threshold = 0.4

outputdirpath = '/home/jean-pierre/Desktop/16feb'
# Set the number of processes you want to use (adjust as needed)
num_processes = 8

curdir = os.getcwd()
divbyframes = 1

#mainpath = os.path.join(curdir,'cut_data')

#fps = 24
#freq = 4
#freq_old = 2

#x=3840
#y=2160

#define base coordinates
Xbase = 2803987
Ybase = 1175193

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





im1 = drone1impath
im2 = drone6impath
im3 = drone16impath

hom1 = drone1hompath
hom2 = drone6hompath
hom3 = drone16hompath

figsize = (80, 40)
cmap = 'Greys'
color = True
n_alpha = 0

fig, ax = plt.subplots(figsize=figsize)


#DRONE6 (2)
drone2_image = cv2.imread(im2)

with open(hom2, 'rb') as pickle_f1:
    drone2_homography = pickle.load(pickle_f1)

X, Y = rectify_image(drone2_image, drone2_homography)

bgrimg = drone2_image
img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

im = ax.pcolormesh(X[:,:], Y[:,:], np.mean(img[:,...], -1), cmap = cmap)

if color:
    rgba = _construct_rgba_vector(img[:,...], n_alpha=n_alpha)
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(rgba)
    #im.set_alpha(opacity)



#DRONE 1 (1)

drone1_image = cv2.imread(im1)

with open(hom1, 'rb') as pickle_f1:
    drone1_homography = pickle.load(pickle_f1)

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







#DRONE 16 (3)

drone3_image = cv2.imread(im3)

with open(hom3, 'rb') as pickle_f1:
    drone3_homography = pickle.load(pickle_f1)

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
plt.title("Surface Velocities Median of 5 Videos")
plt.text(190, 17, str(threshold_value)+' m/s')


#df = pd.read_csv('/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/merged_data_velocities_2.csv')
df_raw = pd.read_csv('/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/merged_data_rotated_2_minus30.csv')

# Filtering rows based on the condition
df = df_raw[df_raw['Norme'] <= threshold_value]

#cmap = plt.get_cmap('viridis')

orange_rgb = (255/255, 165/255, 0/255)
purple_rgb = (128/255, 0/255, 128/255)

colors = [purple_rgb, orange_rgb]
#cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
# Create a colormap with two segments
#segments = [0, 4, 6]
#map = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)



def custom_colormap(color1, color2):
    # Define color points and corresponding colors
    cdict = {'red':   [(0.0, color1[0], color1[0]),
                       (4.0 / 6.0, color2[0], color2[0]),
                       (1.0, color2[0], color2[0])],
             'green': [(0.0, color1[1], color1[1]),
                       (4.0 / 6.0, color2[1], color2[1]),
                       (1.0, color2[1], color2[1])],
             'blue':  [(0.0, color1[2], color1[2]),
                       (4.0 / 6.0, color2[2], color2[2]),
                       (1.0, color2[2], color2[2])]}
    
    # Create and return the colormap
    return LinearSegmentedColormap('custom_colormap', cdict)

cmap = custom_colormap(purple_rgb,orange_rgb)


df['V'] = np.sqrt(df['Vx']**2 + df['Vy']**2)
df['normalized_V'] = (df['V'] - df['V'].min()) / (df['V'].max() - df['V'].min())

# Extract coordinates and velocities
X = df['X']
Y = df['Y']
Vx = df['Vx']
Vy = df['Vy']

# Calculate arrow lengths (magnitudes of velocities)
arrow_lengths = (Vx**2 + Vy**2)**0.5

# Normalize arrow lengths to fit the range [0, 0.5]
normalized_arrow_lengths = arrow_lengths / arrow_lengths.max()# * 0.5
#max_arrow_length = arrow_lengths.max()
#map = plt.get_cmap('viridis')  #coolwarm You can choose a different colormap

# Plot arrows using quiver
quiver = ax.quiver(X, Y, Vx, Vy, scale=1, scale_units='xy', angles='xy', width=0.002, color=cmap(normalized_arrow_lengths))#3, color=cmap(normalized_arrow_lengths))




# Add colorbar below the image
norm = plt.Normalize(df['V'].min(), df['V'].max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(df['V'])

# Create a new axes for the colorbar
cbar_ax = plt.gcf().add_axes([0.5, 0.1, 0.4, 0.04])  # Adjust the values as needed

# Add colorbar for reference below the image
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Velocity (m/s)')

# Ensure the colorbar is updated with correct colors
#sm._A = []



#plt.ylim(1175215-Ybase, 1175295-Ybase)
#plt.xlim(2804000-Xbase, 2804190-Xbase)












plt.tight_layout()
plt.savefig(outputimpath)

fig.clf()
plt.close(fig)
plt.close()
plt.clf()
del X, Y
gc.collect()