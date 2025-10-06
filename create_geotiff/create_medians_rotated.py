import pandas as pd

file1 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid1/all_vid1_rotated.csv'
file2 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid2/all_vid2_rotated.csv'
file3 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid3/all_vid3_rotated.csv'
file4 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid4/all_vid4_rotated.csv'
file5 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid5/all_vid5_rotated.csv'

# List of file names
file_names = [file1,file2,file3,file4,file5]

# Read each CSV file into a DataFrame and set 'X' and 'Y' columns as the index
dataframes = [pd.read_csv(file).set_index(['X', 'Y']) for file in file_names]

# Merge DataFrames based on 'X' and 'Y' coordinates
merged_df = pd.concat(dataframes).groupby(['X', 'Y']).median().reset_index()

# Set values smaller than 0.001 to 0
merged_df[['Vx', 'Vy']] = merged_df[['Vx', 'Vy']].applymap(lambda x: 0 if abs(x) < 0.001 else x)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data_rotated.csv', index=False)