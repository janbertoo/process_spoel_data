import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from shapely.geometry import LineString
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial.distance import cdist

savefigfolder = 'step9_dataframe/'

df = pd.read_pickle(savefigfolder+'dataframe_vidALL_filtered_no_drone_overlap.p')
print(df)
def project_point_on_line(point,line):
    
    dist = line.project(point)

    return(dist)


def calculate_speed(row):
    #print(np.isnan(row['center_geo'][0]))
    #print(np.isnan(row['center_before']))
    # Check if any value in center_geo or center_before is NaN
    #if row['center_geo'] == 'nan' or row['center_before'] == 'nan':
        #return np.nan
    
    try:
        # Convert string representation of coordinates to lists of floats
        center_geo = row['center_geo']
        center_before = row['center_before']

        # Extracting x and y coordinates from lists
        x1, y1 = center_geo
        x2, y2 = center_before

        # Calculating distance using Euclidean distance formula
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Calculating time difference
        time_difference = row['timestamp'] - row['timestamp_before']

        # Calculating speed
        speed = distance / time_difference

        return speed

    except:
        return np.nan

def calculate_speed_x(row):
    try:
        # Convert string representation of coordinates to lists of floats
        center_geo = row['center_geo']
        center_before = row['center_before']

        # Extracting x and y coordinates from lists
        x1, y1 = center_geo
        x2, y2 = center_before

        # Calculating distance using Euclidean distance formula
        distance = (x1 - x2)

        # Calculating time difference
        time_difference = row['timestamp'] - row['timestamp_before']

        # Calculating speed
        speed_x = distance / time_difference

        return speed_x

    except:
        return np.nan

def calculate_speed_y(row):
    try:
        # Convert string representation of coordinates to lists of floats
        center_geo = row['center_geo']
        center_before = row['center_before']

        # Extracting x and y coordinates from lists
        x1, y1 = center_geo
        x2, y2 = center_before

        # Calculating distance using Euclidean distance formula
        distance = (y1 - y2)

        # Calculating time difference
        time_difference = row['timestamp'] - row['timestamp_before']

        # Calculating speed
        speed_y = distance / time_difference

        return speed_y

    except:
        return np.nan

def calculate_rotation(row):
    #print(np.isnan(row['center_geo'][0]))
    #print(np.isnan(row['center_before']))
    # Check if any value in center_geo or center_before is NaN
    #if row['center_geo'] == 'nan' or row['center_before'] == 'nan':
        #return np.nan
    
    try:
        # Convert string representation of coordinates to lists of floats
        tl = row['']
        bl = row['']
        br = row['']

        # Extracting x and y coordinates from lists
        x_tl, y_tl = tl
        x_bl, y_bl = bl
        x_br, y_br = br

        left = np.abs( y_tl - y_bl )
        bottom = np.abs( x_bl - x_br )

        # Calculating time difference
        time_difference = row['timestamp'] - row['timestamp_before']

        # Calculating speed
        speed = distance / time_difference

        return speed

    except:
        return np.nan



#vidnumber = 5
#df = pd.read_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'.p')
#df = df_all[df_all['vidnumber'] == vidnumber]

#.read_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'.p')


# Assuming your DataFrame is already defined as 'df'

pd.to_numeric(df['timestamp'])
# Sort DataFrame by 'ID' and 'timestamp'
df_sorted = df.sort_values(['vidnumber', 'ID', 'timestamp'])

# Calculate 'center_before' and 'timestamp_before' by shifting values
df_sorted['center_before'] = df_sorted.groupby(['vidnumber','ID'])['center_geo'].shift()
df_sorted['timestamp_before'] = df_sorted.groupby(['vidnumber','ID'])['timestamp'].shift()
df_sorted['frame_number_before'] = df_sorted.groupby(['vidnumber','ID'])['frame_number'].shift()
#df_sorted['iloc_before'] = df_sorted.groupby('ID')['frame_number'].shift()

df_sorted['width_before'] = df_sorted.groupby(['vidnumber','ID'])['width'].shift()
df_sorted['height_before'] = df_sorted.groupby(['vidnumber','ID'])['height'].shift()

# Calculate time difference in seconds
#df_sorted['timestamp_before'] = (df_sorted['timestamp'] - df_sorted['timestamp_before'])

# Print the DataFrame with the new columns
#print(df_sorted)

df_sorted['center_geo'] = df_sorted['center_geo'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else np.nan)
df_sorted['center_before'] = df_sorted['center_before'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else np.nan)

df_sorted['time_difference'] = df_sorted['timestamp'] - df_sorted['timestamp_before']


#df_sorted.reindex()
#df_sorted = df_sorted.reset_index(drop=True)

df_sorted = df_sorted[df_sorted['time_difference'] != 0]
df_sorted = df_sorted.reset_index(drop=True)

df_sorted['speed'] = df_sorted.apply(calculate_speed, axis=1)
df_sorted['speed_x'] = df_sorted.apply(calculate_speed_x, axis=1)
df_sorted['speed_y'] = df_sorted.apply(calculate_speed_y, axis=1)

#print(df_sorted['center_geo_x'])
#print(df_sorted['center_geo_y'])

df_sorted.to_pickle(savefigfolder+'dataframe_vidALL_filtered_no_drone_overlap_shifted.p')


































