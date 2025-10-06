import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from tabulate import tabulate

savefigfolder = 'step9_dataframe/'

#delay_frames = [30,0,115] #vid3
#delay_frames = [24,0,22] #vid4

#delay_frames = [50, 0, 327] #vid1
#delay_frames = [60,0,211] #vid2
#delay_frames = [67,0,152] #vid3
#delay_frames = [0,4,33] #vid4
#delay_frames = [0,91,1] #vid5

alldata = [
[1, [50, 0, 327] ],
[2, [60,0,211] ],
[3, [67,0,152] ],
[4, [0,4,33] ],
[5, [0,91,1] ]
]

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


all_dfs = []
all_dfs_filtered = []

for vidnumber, delay_frames in alldata:

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

    # Load data into dataframes
    dfs = []
    for i, folder_path in enumerate(folder_paths):
        df = load_data(folder_path, drone_numbers[i], delay_frames[i])
        dfs.append(df)

    # Concatenate dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Calculate timestamp based on frame number, delay frames, and sampling rate
    frame_rate = 24  # Hz
    sampling_rate = 4  # frames per second
    combined_df['timestamp'] = (combined_df['frame_number'] + combined_df['delay_frames'] * ( frame_rate / sampling_rate ) ) / frame_rate

    df = combined_df

    df['vidnumber'] = vidnumber

    
    # Create a new dataframe with selected columns
    selected_columns_df = combined_df[['ID', 'length', 'diameter']]

    # Group by 'ID' and calculate the median for each group
    median_values_df = selected_columns_df.groupby('ID').median()
    median_values_df['volume'] = median_values_df['length'] * np.pi * (median_values_df['diameter']/2) ** 2

    # Convert ID index to numerical values
    median_values_df.index = median_values_df.index.astype(int)

    # Sort the dataframe by the ID index
    median_values_df = median_values_df.sort_index()







    ######MERGE df and median
    df_to_save = df
    # Convert the 'ID' column to the same data type in both dataframes
    df_to_save['ID'] = df_to_save['ID'].astype(int)
    median_values_df.index = median_values_df.index.astype(int)

    # Merge the dataframes on the 'ID' column and rename the columns
    merged_df_unfiltered = pd.merge(df_to_save, median_values_df, on='ID', how='left', suffixes=('', '_median'))

    # Save the merged df to a pickle file
    merged_df_unfiltered.to_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'.p')
    


    all_dfs.append(merged_df_unfiltered.copy())


    ################################################################################# REMOVE DOUBLE DATA OF OVERLAPPING DRONES

    # Convert 'center_geo' column to numeric values using ast.literal_eval()
    merged_df_unfiltered['center_geo_x'] = merged_df_unfiltered['center_geo'].apply(lambda x: ast.literal_eval(x)[0])
    merged_df_unfiltered['center_geo_y'] = merged_df_unfiltered['center_geo'].apply(lambda x: ast.literal_eval(x)[1])

    # Filter for drone number 1 and geo_center is right of line 1
    line1_condition = (merged_df_unfiltered['drone_number'] == 1) & \
                      (((merged_df_unfiltered['center_geo_x'] - 2) * m_line1 + b_line1) < merged_df_unfiltered['center_geo_y'])
    #print("Line 1 condition:\n", line1_condition)
    #print("Filtered DataFrame with line 1 condition:\n", merged_df_unfiltered[line1_condition])

    # Filter for drone number 6 and geo_center is left of line 1 and right of line 2
    line2_condition = (merged_df_unfiltered['drone_number'] == 6) & \
                      (((merged_df_unfiltered['center_geo_x'] - 2) * m_line1 + b_line1) > merged_df_unfiltered['center_geo_y']) & \
                      (((merged_df_unfiltered['center_geo_x'] - 2) * m_line2 + b_line2) < merged_df_unfiltered['center_geo_y'])
    #print("Line 2 condition:\n", line2_condition)
    #print("Filtered DataFrame with line 2 condition:\n", merged_df_unfiltered[line2_condition])

    # Filter for drone number 16 and geo_center is left of line 2
    line3_condition = (merged_df_unfiltered['drone_number'] == 16) & \
                      (((merged_df_unfiltered['center_geo_x'] - 2) * m_line2 + b_line2) > merged_df_unfiltered['center_geo_y'])
    #print("Line 3 condition:\n", line3_condition)
    #print("Filtered DataFrame with line 3 condition:\n", merged_df_unfiltered[line3_condition])

    # Apply the conditions using logical OR to keep rows that meet any of the conditions
    filtered_df = merged_df_unfiltered[line1_condition | line2_condition | line3_condition]
    




    ################################################################################# PLOT THE DIFFERENCE BETWEEN THE UNFILTRED AND THE FILTERED DATA

    # Plot all points
    plt.figure(figsize=(20, 10))
    plt.scatter(merged_df_unfiltered['center_geo_x'], merged_df_unfiltered['center_geo_y'], marker='o', color='blue', label='All Points')
    plt.scatter(filtered_df['center_geo_x'], filtered_df['center_geo_y'], marker='o', color='red', label='All Points')

    # Add labels and title
    plt.xlabel('X Center')
    plt.ylabel('Y Center')
    plt.title('Plot of All Points')
    plt.legend()

    # Show plot
    plt.grid(True)
    #plt.savefig(savefigfolder+'unfiltered_and_filtered.png')
    plt.savefig(savefigfolder+'unfiltered_and_filtered_vid'+str(vidnumber)+'.png')
    #plt.savefig(savefigfolder+'unfiltered_and_filtered_vid'+str(vidnumber)+'.eps')


    ################################################################################# SAVE DATA



    # Convert the 'ID' column to the same data type in both dataframes
    filtered_df['ID'] = filtered_df['ID'].astype(int)
    #median_values_df.index = median_values_df.index.astype(int)

    # Merge the dataframes on the 'ID' column and rename the columns
    #merged_df = pd.merge(filtered_df, median_values_df, on='ID', how='left', suffixes=('', '_median'))

    # Save the merged df to a pickle file
    filtered_df.to_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'_filtered_no_drone_overlap.p')

    print('LENGTH')
    print(len(median_values_df))
    print(median_values_df)
    print('')

    all_dfs_filtered.append(filtered_df.copy())




df = pd.concat(all_dfs, ignore_index=True)
df_filtered = pd.concat(all_dfs_filtered, ignore_index=True)


# Save the merged df to a pickle file
#df.to_pickle(savefigfolder+'dataframe_vidALL_filtered_no_drone_overlap.p')
#df_filtered.to_pickle(savefigfolder+'dataframe_vidALL.p')

# Save the merged df to a pickle file
df_filtered.to_pickle(savefigfolder+'dataframe_vidALL_filtered_no_drone_overlap.p')
df.to_pickle(savefigfolder+'dataframe_vidALL.p')


