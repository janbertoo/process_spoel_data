import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from shapely.geometry import LineString
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial.distance import cdist

savefigfolder = 'step9_dataframe/'
vidnumber = 5

# Define base coordinates
Xbase = 2803987
Ybase = 1175193

Section1 = ((2804162.1299,1175290.8655),(2804181.03779,1175253.16418))
Section2 = ((2804132.9050,1175280.4683),(2804142.0906,1175249.4849))
Section3 = ((2804104.0364,1175274.5377),(2804106.5156,1175248.0643))
Section4 = ((2804056.8109,1175273.5760),(2804070.4716,1175236.7660))
Section5 = ((2804033.1186,1175264.8710),(2804059.1070,1175228.7094))
Section6 = ((2804005.0938,1175250.3824),(2804041.5581,1175221.2789))

#create 3 line segments for the sections
lineSections = [LineString([Section1[0], Section1[1]]),LineString([Section2[0], Section2[1]]),LineString([Section3[0], Section3[1]]),LineString([Section4[0], Section4[1]]),LineString([Section5[0], Section5[1]]),LineString([Section6[0], Section6[1]])]

df = pd.read_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'.p')
#print(df)

water_levels = [1487.88, 1488.2, 1488.662, 1488.95, 1489.55, 1490.03]

pre_data_files = [savefigfolder+'presection1.csv',savefigfolder+'presection2.csv',savefigfolder+'presection3.csv',savefigfolder+'presection4.csv',savefigfolder+'presection5.csv',savefigfolder+'presection6.csv']
post_data_files = [savefigfolder+'postsection1.csv',savefigfolder+'postsection2.csv',savefigfolder+'postsection3.csv',savefigfolder+'postsection4.csv',savefigfolder+'postsection5.csv',savefigfolder+'postsection6.csv']

#dist = lineSectionAAnorthSouth.project(point)


def combine_dataframes(savefigfolder, vidnumbers):
    dataframes = []

    for vidnumber in vidnumbers:
        # Construct file path
        file_path = savefigfolder + 'dataframe_vid' + str(vidnumber) + '.p'
        
        # Read the dataframe
        df = pd.read_pickle(file_path)
        
        # Add a new column with the vidnumber
        df['vidnumber'] = vidnumber
        
        # Add the dataframe to the list
        dataframes.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

# Define savefigfolder and vidnumbers
#savefigfolder = 'path_to_folder/'  # Replace with your folder path
vidnumbers = [1, 2, 3,4,5]  # List of vidnumbers

# Combine dataframes and get the combined dataframe
combined_df = combine_dataframes(savefigfolder, vidnumbers)
#df = combined_df
print(combined_df)

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



dataframes = []

for vidnumber in vidnumbers:

#vidnumber = 5
    df = pd.read_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'.p')


    # Assuming your DataFrame is already defined as 'df'

    pd.to_numeric(df['timestamp'])
    # Sort DataFrame by 'ID' and 'timestamp'
    df_sorted = df.sort_values(['ID', 'timestamp'])

    # Calculate 'center_before' and 'timestamp_before' by shifting values
    df_sorted['center_before'] = df_sorted.groupby('ID')['center_geo'].shift()
    df_sorted['timestamp_before'] = df_sorted.groupby('ID')['timestamp'].shift()
    df_sorted['frame_number_before'] = df_sorted.groupby('ID')['frame_number'].shift()
    #df_sorted['iloc_before'] = df_sorted.groupby('ID')['frame_number'].shift()

    df_sorted['width_before'] = df_sorted.groupby('ID')['width'].shift()
    df_sorted['height_before'] = df_sorted.groupby('ID')['height'].shift()

    # Calculate time difference in seconds
    #df_sorted['timestamp_before'] = (df_sorted['timestamp'] - df_sorted['timestamp_before'])

    # Print the DataFrame with the new columns
    print(df_sorted)

    df_sorted['center_geo'] = df_sorted['center_geo'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else np.nan)
    df_sorted['center_before'] = df_sorted['center_before'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else np.nan)

    df_sorted['time_difference'] = df_sorted['timestamp'] - df_sorted['timestamp_before']



    print(df_sorted)

    df_sorted['speed'] = df_sorted.apply(calculate_speed, axis=1)
    df_sorted['speed_x'] = df_sorted.apply(calculate_speed_x, axis=1)
    df_sorted['speed_y'] = df_sorted.apply(calculate_speed_y, axis=1)

    print(df_sorted['center_geo_x'])
    print(df_sorted['center_geo_y'])



    # Filter the dataframe based on the condition
    threshold_value = 2  # Set your desired threshold value here
    filtered_df_below_1sec_time_diff = df_sorted[df_sorted['time_difference'] < threshold_value]

    #filtered_df_below_1sec_time_diff = filtered_df_below_1sec_time_diff
    filtered_df_below_1sec_time_diff = filtered_df_below_1sec_time_diff[filtered_df_below_1sec_time_diff['drone_number'].isin([1, 6])]

    # Print the filtered dataframe
    print(filtered_df_below_1sec_time_diff)

    LSPIV_speeds_path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/step6_LSPIV/all_vid'+str(vidnumber)+'_05.csv'

    column_names = ['center_geo_x', 'center_geo_y', 'speed_x','speed_y']  # Add more column names as needed
    df_LSPIV_speeds = pd.read_csv(LSPIV_speeds_path, names=column_names)

    print(df_LSPIV_speeds)





    # Calculate absolute speed for the LSPIV dataframe
    df_LSPIV_speeds['speed_LSPIV'] = np.sqrt(df_LSPIV_speeds['speed_x']**2 + df_LSPIV_speeds['speed_y']**2)


    # Reset index of df_LSPIV_speeds to avoid duplicate index labels
    df_LSPIV_speeds_reset = df_LSPIV_speeds.reset_index(drop=True)

    # Calculate pairwise distances between all points in both dataframes
    distances = cdist(filtered_df_below_1sec_time_diff[['center_geo_x', 'center_geo_y']],
                      df_LSPIV_speeds_reset[['center_geo_x', 'center_geo_y']])

    # Find the index of the closest point in df_LSPIV_speeds_reset for each point in filtered_df_below_1sec_time_diff
    closest_idx = np.argmin(distances, axis=1)

    # Retrieve the corresponding speeds from df_LSPIV_speeds_reset
    closest_speeds = df_LSPIV_speeds_reset.iloc[closest_idx]

    # Add the LSPIV speeds to filtered_df_below_1sec_time_diff
    filtered_df_below_1sec_time_diff['speed_x_LSPIV'] = closest_speeds['speed_x'].values
    filtered_df_below_1sec_time_diff['speed_y_LSPIV'] = closest_speeds['speed_y'].values
    filtered_df_below_1sec_time_diff['speed_LSPIV'] = closest_speeds['speed_LSPIV'].values

    filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'] = filtered_df_below_1sec_time_diff['speed_LSPIV'] - filtered_df_below_1sec_time_diff['speed']

    filtered_df_below_1sec_time_diff['unique_ID'] = filtered_df_below_1sec_time_diff['ID']+(vidnumber*1000)
    filtered_df_below_1sec_time_diff['vidnumber'] = vidnumber

    print(filtered_df_below_1sec_time_diff)
    dataframes.append(filtered_df_below_1sec_time_diff)

    #determine max volume and min LSPIV_speed_minus_speed
    # Filter the dataframe based on the condition
    #smaller_df = filtered_df_below_1sec_time_diff[filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'] < -7]

    # Display the new, smaller dataframe
    #print(smaller_df)
    #filtered_df_below_1sec_time_diff = smaller_df

    '''

    # Filter the dataframe based on the condition
    smaller_df = filtered_df_below_1sec_time_diff[filtered_df_below_1sec_time_diff['volume_median'] < 0.22]

    # Display the new, smaller dataframe
    print(smaller_df)
    filtered_df_below_1sec_time_diff = smaller_df

    # Filter the dataframe based on the condition
    smaller_df = filtered_df_below_1sec_time_diff[filtered_df_below_1sec_time_diff['speed'] < 7]

    # Display the new, smaller dataframe
    print(smaller_df)
    filtered_df_below_1sec_time_diff = smaller_df

    '''
















import numpy as np
import matplotlib.pyplot as plt

# Convert columns to numeric data type
filtered_df_below_1sec_time_diff['length_median'] = pd.to_numeric(filtered_df_below_1sec_time_diff['length_median'], errors='coerce')
filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'] = pd.to_numeric(filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'], errors='coerce')

# Remove rows with NaN or infinite values
filtered_df_below_1sec_time_diff = filtered_df_below_1sec_time_diff.replace([np.inf, -np.inf], np.nan)
filtered_df_below_1sec_time_diff.dropna(subset=['length_median', 'LSPIV_speed_minus_speed'], inplace=True)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(20,10))

# Plot 'speed' column as dots
ax.scatter(filtered_df_below_1sec_time_diff['length_median'], filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'], label='Speed', marker='.')

# Perform linear regression only on non-NaN values
x = filtered_df_below_1sec_time_diff['length_median']
y = filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed']

# Check if there are any NaN values in 'x' or 'y'
print("NaN in x:", np.isnan(x).any())
print("NaN in y:", np.isnan(y).any())

# If there are no NaN values, perform linear regression
if not np.isnan(x).any() and not np.isnan(y).any():
    m, b = np.polyfit(x, y, 1)
    print("Slope:", m)
    print("Intercept:", b)
    
    # Plot the line of best fit
    ax.plot(x, m*x + b, color='red', label='Line of Best Fit, Slope: '+str(m)+', Intercept: '+str(b))
else:
    print("Error: NaN values present in data.")

# Set labels and title
ax.set_xlabel('Length Median')
ax.set_ylabel('Speed LSPIV minus speed logs')
ax.set_title('Speed vs Length Median')
ax.legend()

#ax.set_xlim([xmin, xmax])
ax.set_ylim([-3, 2])

# Show plot
#plt.show()
plt.savefig(savefigfolder+'length_median_vid'+str(vidnumber)+'.png')
plt.savefig(savefigfolder+'length_median_vid'+str(vidnumber)+'.svg')




import numpy as np
import matplotlib.pyplot as plt

# Convert columns to numeric data type
filtered_df_below_1sec_time_diff['diameter_median'] = pd.to_numeric(filtered_df_below_1sec_time_diff['diameter_median'], errors='coerce')
filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'] = pd.to_numeric(filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'], errors='coerce')

# Remove rows with NaN or infinite values
filtered_df_below_1sec_time_diff = filtered_df_below_1sec_time_diff.replace([np.inf, -np.inf], np.nan)
filtered_df_below_1sec_time_diff.dropna(subset=['diameter_median', 'LSPIV_speed_minus_speed'], inplace=True)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(20,10))

# Plot 'speed' column as dots
ax.scatter(filtered_df_below_1sec_time_diff['diameter_median'], filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'], label='Speed', marker='.')

# Perform linear regression only on non-NaN values
x = filtered_df_below_1sec_time_diff['diameter_median']
y = filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed']

# If there are no NaN values, perform linear regression
if not np.isnan(x).any() and not np.isnan(y).any():
    m, b = np.polyfit(x, y, 1)
    print("Slope:", m)
    print("Intercept:", b)
    
    # Plot the line of best fit
    ax.plot(x, m*x + b, color='red', label='Line of Best Fit, Slope: '+str(m)+', Intercept: '+str(b))
else:
    print("Error: NaN values present in data.")

# Set labels and title
ax.set_xlabel('Diameter Median')
ax.set_ylabel('Speed LSPIV minus speed logs')
ax.set_title('Speed vs Diameter Median')
ax.legend()

#ax.set_xlim([xmin, xmax])
ax.set_ylim([-3, 2])

# Show plot
#plt.show()
plt.savefig(savefigfolder+'diameter_median_vid'+str(vidnumber)+'.png')
plt.savefig(savefigfolder+'diameter_median_vid'+str(vidnumber)+'.svg')



import numpy as np
import matplotlib.pyplot as plt

# Convert columns to numeric data type
filtered_df_below_1sec_time_diff['volume_median'] = pd.to_numeric(filtered_df_below_1sec_time_diff['volume_median'], errors='coerce')
filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'] = pd.to_numeric(filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'], errors='coerce')

# Remove rows with NaN or infinite values
filtered_df_below_1sec_time_diff = filtered_df_below_1sec_time_diff.replace([np.inf, -np.inf], np.nan)
filtered_df_below_1sec_time_diff.dropna(subset=['volume_median', 'LSPIV_speed_minus_speed'], inplace=True)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(20,10))

# Plot 'speed' column as dots
ax.scatter(filtered_df_below_1sec_time_diff['volume_median'], filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed'], label='Speed', marker='.')

# Perform linear regression only on non-NaN values
x = filtered_df_below_1sec_time_diff['volume_median']
y = filtered_df_below_1sec_time_diff['LSPIV_speed_minus_speed']

# If there are no NaN values, perform linear regression
if not np.isnan(x).any() and not np.isnan(y).any():
    m, b = np.polyfit(x, y, 1)
    print("Slope:", m)
    print("Intercept:", b)
    
    # Plot the line of best fit
    ax.plot(x, m*x + b, color='red', label='Line of Best Fit, Slope: '+str(m)+', Intercept: '+str(b))
else:
    print("Error: NaN values present in data.")



# Set labels and title
ax.set_xlabel('Volume Median')
ax.set_ylabel('Speed LSPIV minus speed logs')
ax.set_title('Speed vs Volume Median')
ax.legend()

#ax.set_xlim([xmin, xmax])
ax.set_ylim([-3, 2])

# Show plot
#plt.show()
plt.savefig(savefigfolder+'volume_median_vid'+str(vidnumber)+'.png')
plt.savefig(savefigfolder+'volume_median_vid'+str(vidnumber)+'.svg')


import numpy as np
import matplotlib.pyplot as plt

# Convert columns to numeric data type
filtered_df_below_1sec_time_diff['volume_median'] = pd.to_numeric(filtered_df_below_1sec_time_diff['volume_median'], errors='coerce')
filtered_df_below_1sec_time_diff['speed'] = pd.to_numeric(filtered_df_below_1sec_time_diff['speed'], errors='coerce')

# Remove rows with NaN or infinite values
filtered_df_below_1sec_time_diff = filtered_df_below_1sec_time_diff.replace([np.inf, -np.inf], np.nan)
filtered_df_below_1sec_time_diff.dropna(subset=['volume_median', 'speed'], inplace=True)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(20,10))

# Plot 'speed' column as dots
ax.scatter(filtered_df_below_1sec_time_diff['volume_median'], filtered_df_below_1sec_time_diff['speed'], label='Speed', marker='.')

# Perform linear regression only on non-NaN values
x = filtered_df_below_1sec_time_diff['volume_median']
y = filtered_df_below_1sec_time_diff['speed']

# If there are no NaN values, perform linear regression
if not np.isnan(x).any() and not np.isnan(y).any():
    m, b = np.polyfit(x, y, 1)
    print("Slope:", m)
    print("Intercept:", b)
    
    # Plot the line of best fit
    ax.plot(x, m*x + b, color='red', label='Line of Best Fit, Slope: '+str(m)+', Intercept: '+str(b))
else:
    print("Error: NaN values present in data.")



# Set labels and title
ax.set_xlabel('Volume Median (m^3)')
ax.set_ylabel('Speed (m/s)')
ax.set_title('Speed vs Volume Median')
ax.legend()

#ax.set_xlim([xmin, xmax])
ax.set_ylim([-3, 2])


print('')
print(len(filtered_df_below_1sec_time_diff))

# Show plot
plt.show()
plt.savefig(savefigfolder+'volume_median_speed_vid'+str(vidnumber)+'.png')
plt.savefig(savefigfolder+'volume_median_speed_vid'+str(vidnumber)+'.svg')

