import pandas as pd

# List of file names
file1 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid1/all_vid1.csv'
file2 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid2/all_vid2.csv'
file3 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid3/all_vid3.csv'
file4 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid4/all_vid4.csv'
file5 = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/velocities_vid5/all_vid5.csv'

# Read the two CSV files into DataFrames
file1 = pd.read_csv(file1, header=None, names=['X', 'Y', 'Vx', 'Vy'])
file2 = pd.read_csv(file5, header=None, names=['X', 'Y', 'Vx', 'Vy'])

# Merge DataFrames based on 'X' and 'Y' coordinates
merged_df = pd.merge(file1, file2, on=['X', 'Y'], suffixes=('_file1', '_file2'))

# Subtract velocities from the second file from velocities of the first file
merged_df['Vx_diff'] = merged_df['Vx_file1'] - merged_df['Vx_file2']
merged_df['Vy_diff'] = merged_df['Vy_file1'] - merged_df['Vy_file2']

# Drop unnecessary columns if needed
merged_df = merged_df[['X', 'Y', 'Vx_diff', 'Vy_diff']]

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('subtracted_data_15.csv', index=False)
