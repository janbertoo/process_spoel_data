import pandas as pd

# Load the 5 CSV files without headers and assign column names
file1 = pd.read_csv('all_vid1_05.csv', header=None, names=['X coordinate', 'Y coordinate', 'X speed', 'Y speed'])
file2 = pd.read_csv('all_vid2_05.csv', header=None, names=['X coordinate', 'Y coordinate', 'X speed', 'Y speed'])
file3 = pd.read_csv('all_vid3_05.csv', header=None, names=['X coordinate', 'Y coordinate', 'X speed', 'Y speed'])
file4 = pd.read_csv('all_vid4_05.csv', header=None, names=['X coordinate', 'Y coordinate', 'X speed', 'Y speed'])
file5 = pd.read_csv('all_vid5_05.csv', header=None, names=['X coordinate', 'Y coordinate', 'X speed', 'Y speed'])

# Merge the files on X and Y coordinates
merged = file1[['X coordinate', 'Y coordinate']].copy()

# Calculate the median of 'X speed' and 'Y speed' across the 5 files
merged['X speed'] = pd.concat([file1['X speed'], file2['X speed'], file3['X speed'], file4['X speed'], file5['X speed']], axis=1).median(axis=1)
merged['Y speed'] = pd.concat([file1['Y speed'], file2['Y speed'], file3['Y speed'], file4['Y speed'], file5['Y speed']], axis=1).median(axis=1)

# Save the result to a new CSV file
merged.to_csv('median_speeds.csv', index=False, header=False)

print("Median speeds file created successfully.")
