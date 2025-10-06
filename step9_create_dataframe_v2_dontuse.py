import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

vidnumber = 2

# Define base coordinates
Xbase = 2803987
Ybase = 1175193

# Define lines
points_line1 = ((2804101.6 - Xbase - 2, 1175278.832 - Ybase), (2804107.518 - Xbase - 2, 1175241.001 - Ybase))
x1_line1, y1_line1 = points_line1[0]
x2_line1, y2_line1 = points_line1[1]
m_line1 = (y1_line1 - y2_line1) / (x1_line1 - x2_line1)
b_line1 = (x1_line1 * y2_line1 - x2_line1 * y1_line1) / (x1_line1 - x2_line1)

points_line2 = ((2804036.78 - Xbase - 2, 1175271.824 - Ybase), (2804069.799 - Xbase - 2, 1175236.847 - Ybase))
x1_line2, y1_line2 = points_line2[0]
x2_line2, y2_line2 = points_line2[1]
m_line2 = (y1_line2 - y2_line2) / (x1_line2 - x2_line2)
b_line2 = (x1_line2 * y2_line2 - x2_line2 * y1_line2) / (x1_line2 - x2_line2)


print("m_line1:", m_line1)
print("b_line1:", b_line1)



# Function to load data from text files into a dataframe
def load_data(folder_path, drone_number, delay_frames):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            frame_number = int(file_name.split('_')[-1].split('.')[0][5:])  # Extract frame number from file name
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip().split()
                    line.insert(5, float(line.pop(5)))  # Convert confidence to float
                    line.append(delay_frames)  # Add delay frames
                    line.append(drone_number)  # Add drone number
                    line.append(frame_number)  # Add frame number
                    data.append(line)
    columns = ['class', 'x_center', 'y_center', 'width', 'height', 'confidence', 'ID', 'orientation', 'center_geo', 
               'top_left_geo', 'top_right_geo', 'bottom_left_geo', 'bottom_right_geo', 
               'length', 'diameter', 'volume', 'delay_frames', 'drone_number', 'frame_number']
    df = pd.DataFrame(data, columns=columns)
    return df

# Folder paths and delay frames
folder_paths = [
    '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodlength_wooddiameter_woodvolume',
    '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone6/drone6vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodlength_wooddiameter_woodvolume',
    '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone16/drone16vid'+str(vidnumber)+'/interpolated_detected_linked_IDs_plus_orientation_coordinates-ce-tl-tr-bl-br_woodlength_wooddiameter_woodvolume',
]

drone_numbers = [1, 6, 16]
delay_frames = [50, 0, 327]

# Load data into dataframes
dfs = []
for i, folder_path in enumerate(folder_paths):
    df = load_data(folder_path, drone_numbers[i], delay_frames[i])
    print("Data type of 'x_coordinate':", df['x_coordinate'].dtype)
    print("Data type of 'y_center':", df['y_center'].dtype)

    
    if 'center_geo' in df.columns:  # Check if 'center_geo' column exists in the DataFrame
        # Extract x-coordinate from 'center_geo' column
        df['x_coordinate'] = df['center_geo'].str.extract(r'\[(.*?),').astype(float)
        
        if drone_numbers[i] == 1:
            # Keep data where geo_center is right of line 1
            df = df[df['x_coordinate'] > (m_line1 * df['y_center'] + b_line1)]
        elif drone_numbers[i] == 6:
            # Keep data where geo_center is left of line 1 and right of line 2
            df = df[(df['x_coordinate'] < (m_line1 * df['y_center'] + b_line1)) &
                    (df['x_coordinate'] > (m_line2 * df['y_center'] + b_line2))]
        elif drone_numbers[i] == 16:
            # Keep data where geo_center is left of line 2
            df = df[df['x_coordinate'] < (m_line2 * df['y_center'] + b_line2)]
        
        dfs.append(df)
    else:
        print(f"Warning: 'center_geo' column not found in DataFrame for drone {drone_numbers[i]}")

# Concatenate dataframes
combined_df = pd.concat(dfs, ignore_index=True)



# Calculate timestamp based on frame number, delay frames, and sampling rate
frame_rate = 24  # Hz
sampling_rate = 4  # frames per second
combined_df['timestamp'] = (combined_df['frame_number'] + combined_df['delay_frames'] * (frame_rate / sampling_rate)) / frame_rate

# Print combined dataframe
print(combined_df)

# Create a new dataframe with selected columns
selected_columns_df = combined_df[['ID', 'length', 'diameter', 'volume']]

# Group by 'ID' and calculate the median for each group
median_values_df = selected_columns_df.groupby('ID').median()

# Convert ID index to numerical values
median_values_df.index = median_values_df.index.astype(int)

# Sort the dataframe by the ID index
median_values_df = median_values_df.sort_index()

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Plot length_median
axes[0].bar(median_values_df.index, median_values_df['length'], color='blue')
axes[0].set_title('Median Length')
axes[0].set_ylabel('Length')
axes[0].axhline(y=1, color='red', linestyle='--')  # Add horizontal line at 1m

# Plot diameter_median
axes[1].bar(median_values_df.index, median_values_df['diameter'], color='orange')
axes[1].set_title('Median Diameter')
axes[1].set_ylabel('Diameter')
axes[1].axhline(y=0.1, color='red', linestyle='--')  # Add horizontal line at 0.
