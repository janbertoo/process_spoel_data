import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from tabulate import tabulate

savefigfolder = 'step9_dataframe/'

#delay_frames = [30,0,115] #vid3
#delay_frames = [24,0,22] #vid4

delay_frames = [50, 0, 327] #vid1
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
    
    # Create a new dataframe with selected columns
    selected_columns_df = combined_df[['ID', 'length', 'diameter']]

    # Group by 'ID' and calculate the median for each group
    median_values_df = selected_columns_df.groupby('ID').median()
    median_values_df['volume'] = median_values_df['length'] * np.pi * (median_values_df['diameter']/2) ** 2
    
    # Convert ID index to numerical values
    median_values_df.index = median_values_df.index.astype(int)

    # Sort the dataframe by the ID index
    median_values_df = median_values_df.sort_index()


    ################################################################################# PLOT LENGTHS AND DIAMETERS OF INDIVIDUAL PIECES
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # Plot length_median
    axes[0].bar(median_values_df.index, median_values_df['length'], color='blue')
    axes[0].set_title('Median Length')
    axes[0].set_ylabel('Length')
    axes[0].axhline(y=1, color='red', linestyle='--')  # Add horizontal line at 1m

    # Plot diameter_median
    axes[1].bar(median_values_df.index, median_values_df['diameter'], color='orange')
    axes[1].set_title('Median Diameter')
    axes[1].set_ylabel('Diameter')
    axes[1].axhline(y=0.1, color='red', linestyle='--')  # Add horizontal line at 0.1m

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.savefig(savefigfolder+'lengths_and_diameters_vid'+str(vidnumber)+'.png')

    # Define bin edges
    length_bins = np.arange(0, median_values_df['length'].max() + 0.25, 0.25)
    diameter_bins = np.arange(0, median_values_df['diameter'].max() + 0.025, 0.025)

    ################################################################################# PLOT LENGTHS AND DIAMETERS IN BINS

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Plot length_median histogram
    axes[0].hist(median_values_df['length'], bins=length_bins, color='blue')
    axes[0].set_title('Median Length Histogram')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=1, color='red', linestyle='--')  # Add vertical line at 1m

    # Plot diameter_median histogram
    axes[1].hist(median_values_df['diameter'], bins=diameter_bins, color='orange')
    axes[1].set_title('Median Diameter Histogram')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0.1, color='red', linestyle='--')  # Add vertical line at 0.1m

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.savefig(savefigfolder+'lengths_and_diameters_bins_vid'+str(vidnumber)+'.png')




    ################################################################################# REMOVE DOUBLE DATA OF OVERLAPPING DRONES

    # Convert 'center_geo' column to numeric values using ast.literal_eval()
    df['center_geo_x'] = df['center_geo'].apply(lambda x: ast.literal_eval(x)[0])
    df['center_geo_y'] = df['center_geo'].apply(lambda x: ast.literal_eval(x)[1])

    # Filter for drone number 1 and geo_center is right of line 1
    line1_condition = (df['drone_number'] == 1) & \
                      (((df['center_geo_x'] - 2) * m_line1 + b_line1) < df['center_geo_y'])
    #print("Line 1 condition:\n", line1_condition)
    #print("Filtered DataFrame with line 1 condition:\n", df[line1_condition])

    # Filter for drone number 6 and geo_center is left of line 1 and right of line 2
    line2_condition = (df['drone_number'] == 6) & \
                      (((df['center_geo_x'] - 2) * m_line1 + b_line1) > df['center_geo_y']) & \
                      (((df['center_geo_x'] - 2) * m_line2 + b_line2) < df['center_geo_y'])
    #print("Line 2 condition:\n", line2_condition)
    #print("Filtered DataFrame with line 2 condition:\n", df[line2_condition])

    # Filter for drone number 16 and geo_center is left of line 2
    line3_condition = (df['drone_number'] == 16) & \
                      (((df['center_geo_x'] - 2) * m_line2 + b_line2) > df['center_geo_y'])
    #print("Line 3 condition:\n", line3_condition)
    #print("Filtered DataFrame with line 3 condition:\n", df[line3_condition])

    # Apply the conditions using logical OR to keep rows that meet any of the conditions
    filtered_df = df[line1_condition | line2_condition | line3_condition]
    




    ################################################################################# PLOT THE DIFFERENCE BETWEEN THE UNFILTRED AND THE FILTERED DATA

    # Plot all points
    plt.figure(figsize=(20, 10))
    plt.scatter(df['center_geo_x'], df['center_geo_y'], marker='o', color='blue', label='All Points')
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


    ################################################################################# SAVE DATA



    # Convert the 'ID' column to the same data type in both dataframes
    filtered_df['ID'] = filtered_df['ID'].astype(int)
    median_values_df.index = median_values_df.index.astype(int)

    # Merge the dataframes on the 'ID' column and rename the columns
    merged_df = pd.merge(filtered_df, median_values_df, on='ID', how='left', suffixes=('', '_median'))

    # Save the merged df to a pickle file
    merged_df.to_pickle(savefigfolder+'dataframe_vid'+str(vidnumber)+'.p')

    print('LENGTH')
    print(len(median_values_df))
    print(median_values_df)
    print('')

    #median_values_df
    smaller_df = median_values_df[median_values_df['diameter'] > 0.10]

    large_wood_df = smaller_df[smaller_df['length'] > 1]

    large_wood_df = large_wood_df.reset_index(drop=True)
    large_wood_df['volume'] = large_wood_df['length'] * np.pi * ( large_wood_df['diameter'] / 2 ) ** 2

    print('Large: '+str(len(large_wood_df)))



    smaller_df = median_values_df[median_values_df['diameter'] <= 0.10]

    semilarge_wood_df_long = smaller_df[smaller_df['length'] > 1]

    semilarge_wood_df_long = semilarge_wood_df_long.reset_index(drop=True)
    semilarge_wood_df_long['volume'] = semilarge_wood_df_long['length'] * np.pi * ( semilarge_wood_df_long['diameter'] / 2 ) ** 2

    print('Long: '+str(len(semilarge_wood_df_long)))


    smaller_df = median_values_df[median_values_df['diameter'] > 0.10]

    semilarge_wood_df_thick = smaller_df[smaller_df['length'] <= 1]

    semilarge_wood_df_thick = semilarge_wood_df_thick.reset_index(drop=True)
    semilarge_wood_df_thick['volume'] = semilarge_wood_df_thick['length'] * np.pi * ( semilarge_wood_df_thick['diameter'] / 2 ) ** 2

    print('Thick: '+str(len(semilarge_wood_df_thick)))


    smaller_df = median_values_df[median_values_df['diameter'] <= 0.10]

    small_wood_df = smaller_df[smaller_df['length'] <= 1]

    small_wood_df = small_wood_df.reset_index(drop=True)
    small_wood_df['volume'] = small_wood_df['length'] * np.pi * ( small_wood_df['diameter'] / 2 ) ** 2

    print('Small: '+str(len(small_wood_df)))


    large_long_thick_small = [
        ['Large',len(large_wood_df),'> 1','> 0.1',large_wood_df['volume'].sum()],
        ['Long',len(semilarge_wood_df_long),'> 1','<= 0.1',semilarge_wood_df_long['volume'].sum()],
        ['Thick',len(semilarge_wood_df_thick),'<= 1','> 0.1',semilarge_wood_df_thick['volume'].sum()],
        ['Small',len(small_wood_df),'<= 1','<= 0.1',small_wood_df['volume'].sum()]
    ]

    print('')
    print('STATS VID '+str(vidnumber))
    print(tabulate(large_long_thick_small, headers=['Wood\nClass', 'Amount', 'Length\n(m)', 'Diameter\n(m)', 'Total Volume\n(m^3)']))
    print('')


'''



median_values_df = pd.concat(all_median_values, ignore_index=True)
filtered_df = pd.concat(all_filtered_dfs, ignore_index=True)
df = pd.concat(all_dfs, ignore_index=True)



# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(30, 15))

# Plot length_median
axes[0].bar(median_values_df.index, median_values_df['length'], color='blue')
axes[0].set_title('Median Length')
axes[0].set_ylabel('Length')
axes[0].axhline(y=1, color='red', linestyle='--')  # Add horizontal line at 1m

# Plot diameter_median
axes[1].bar(median_values_df.index, median_values_df['diameter'], color='orange')
axes[1].set_title('Median Diameter')
axes[1].set_ylabel('Diameter')
axes[1].axhline(y=0.1, color='red', linestyle='--')  # Add horizontal line at 0.1m

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(savefigfolder+'lengths_and_diameters_vidALL.png')

#print('#################################################################################################################')
#print(median_values_df['diameter'].max())
#print(' ')
#print(median_values_df['diameter'])
#print('#################################################################################################################')
# Define bin edges
length_bins = np.arange(0, median_values_df['length'].max() + 0.25, 0.25)
diameter_bins = np.arange(0, median_values_df['diameter'].max() + 0.025, 0.025)

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Plot length_median histogram
axes[0].hist(median_values_df['length'], bins=length_bins, color='blue')
axes[0].set_title('Median Length Histogram')
axes[0].set_ylabel('Frequency')
axes[0].axvline(x=1, color='red', linestyle='--')  # Add vertical line at 1m

# Plot diameter_median histogram
axes[1].hist(median_values_df['diameter'], bins=diameter_bins, color='orange')
axes[1].set_title('Median Diameter Histogram')
axes[1].set_ylabel('Frequency')
axes[1].axvline(x=0.1, color='red', linestyle='--')  # Add vertical line at 0.1m

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(savefigfolder+'lengths_and_diameters_bins_vidALL.png')


import ast

# Convert 'center_geo' column to numeric values using ast.literal_eval()
df['center_geo_x'] = df['center_geo'].apply(lambda x: ast.literal_eval(x)[0])
df['center_geo_y'] = df['center_geo'].apply(lambda x: ast.literal_eval(x)[1])

# Filter for drone number 1 and geo_center is right of line 1
line1_condition = (df['drone_number'] == 1) & \
                  (((df['center_geo_x'] - 2) * m_line1 + b_line1) < df['center_geo_y'])
#print("Line 1 condition:\n", line1_condition)
#print("Filtered DataFrame with line 1 condition:\n", df[line1_condition])

# Filter for drone number 6 and geo_center is left of line 1 and right of line 2
line2_condition = (df['drone_number'] == 6) & \
                  (((df['center_geo_x'] - 2) * m_line1 + b_line1) > df['center_geo_y']) & \
                  (((df['center_geo_x'] - 2) * m_line2 + b_line2) < df['center_geo_y'])
#print("Line 2 condition:\n", line2_condition)
#print("Filtered DataFrame with line 2 condition:\n", df[line2_condition])

# Filter for drone number 16 and geo_center is left of line 2
line3_condition = (df['drone_number'] == 16) & \
                  (((df['center_geo_x'] - 2) * m_line2 + b_line2) > df['center_geo_y'])
#print("Line 3 condition:\n", line3_condition)
#print("Filtered DataFrame with line 3 condition:\n", df[line3_condition])

# Apply the conditions using logical OR to keep rows that meet any of the conditions
filtered_df = df[line1_condition | line2_condition | line3_condition]
#print("Filtered DataFrame after combining conditions:\n", filtered_df)




#print(filtered_df)










import matplotlib.pyplot as plt

# Plot all points
plt.figure(figsize=(20, 10))
plt.scatter(df['center_geo_x'], df['center_geo_y'], marker='o', color='blue', label='All Points')
plt.scatter(filtered_df['center_geo_x'], filtered_df['center_geo_y'], marker='o', color='red', label='All Points')

# Add labels and title
plt.xlabel('X Center')
plt.ylabel('Y Center')
plt.title('Plot of All Points')
plt.legend()

# Show plot
plt.grid(True)
#plt.savefig(savefigfolder+'unfiltered_and_filtered.png')
plt.savefig(savefigfolder+'unfiltered_and_filtered_vidALL.png')




#merged_df = pd.merge(combined_df, median_values_df, on='ID', how='left')
#merged_df = pd.merge(combined_df, median_values_df, on='ID', how='left', suffixes=('', '_median'))
#print(merged_df)



# Convert the 'ID' column to the same data type in both dataframes
filtered_df['ID'] = filtered_df['ID'].astype(int)
median_values_df.index = median_values_df.index.astype(int)


print(filtered_df['ID'])
print(median_values_df['ID'])

# Merge the dataframes on the 'ID' column and rename the columns
merged_df = pd.merge(filtered_df, median_values_df, on='ID', how='left', suffixes=('', '_median'))

# Print the merged dataframe to verify the changes
#print(merged_df)


merged_df.to_pickle(savefigfolder+'dataframe_vidALL.p')


'''

