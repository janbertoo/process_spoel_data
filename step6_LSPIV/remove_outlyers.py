import pandas as pd
import numpy as np
from scipy.spatial import KDTree



csvs = [
    'all_vid1_05',
    'all_vid2_05',
    'all_vid3_05',
    'all_vid4_05',
    'all_vid5_05',
    ]


for csv_file in csvs:

    # Load the data, ensuring numeric types
    data = pd.read_csv(csv_file+'.csv', header=None, names=['X', 'Y', 'speed_X', 'speed_Y'], dtype={'X': float, 'Y': float, 'speed_X': float, 'speed_Y': float})

    # Calculate the speed magnitude for each point (not saved in the final file)
    data['speed'] = np.sqrt(data['speed_X']**2 + data['speed_Y']**2)

    # Define a function to identify outliers based on z-score
    def identify_outliers(data, z_thresh=3): #is 9 for no filter
        mean_speed = np.mean(data['speed'])
        std_speed = np.std(data['speed'])
        
        # Z-score calculation: (value - mean) / std deviation
        data['z_score'] = (data['speed'] - mean_speed) / std_speed
        
        # Output the number of outliers
        outliers_mask = data['z_score'].abs() > z_thresh
        outliers_count = outliers_mask.sum()
        print(f"Number of outliers found with z_thresh={z_thresh}: {outliers_count}")
        
        # Print the X and Y coordinates of the outliers
        if outliers_count > 0:
            outliers_data = data[outliers_mask][['X', 'Y', 'speed', 'z_score']]
            print("Outliers found at the following coordinates (X, Y, Speed, Z-score):")
            for idx, row in outliers_data.iterrows():
                print(f"X: {row['X']}, Y: {row['Y']}, Speed: {row['speed']}, Z-score: {row['z_score']}")
        
        return outliers_mask

    # Create a KDTree for fast nearest-neighbor lookup
    coordinates = data[['X', 'Y']].values
    tree = KDTree(coordinates)

    # Function to find the average of surrounding cells within a radius
    def get_neighbors_average(data, index, tree, radius=1.0):
        current_coords = data.iloc[index][['X', 'Y']].values
        # Query the KDTree to find all points within the radius
        indices = tree.query_ball_point(current_coords, r=radius)
        
        if len(indices) > 1:  # If there are neighbors other than the point itself
            neighbors_speed = data.iloc[indices]['speed']
            # Return the average speed of the neighbors
            return neighbors_speed.mean()
        return np.nan  # In case no valid neighbors are found

    # Identify outliers based on z-score
    outliers = identify_outliers(data)

    # Replace outliers with the average speed of surrounding cells
    for i in data[outliers].index:
        avg_speed = get_neighbors_average(data, i, tree)
        if not np.isnan(avg_speed):
            data.at[i, 'speed'] = avg_speed
            # Adjust the speed_X and speed_Y components proportionally to the new speed
            norm_factor = avg_speed / np.sqrt(data.at[i, 'speed_X']**2 + data.at[i, 'speed_Y']**2)
            data.at[i, 'speed_X'] *= norm_factor
            data.at[i, 'speed_Y'] *= norm_factor

    # Remove any extra columns and save the cleaned data to a new CSV file
    cleaned_data = data[['X', 'Y', 'speed_X', 'speed_Y']].copy()  # Ensure only original columns are kept
    cleaned_data.to_csv(csv_file+'_no_outliers.csv', index=False, float_format='%.8f',header=False) # Ensure output format is identical
