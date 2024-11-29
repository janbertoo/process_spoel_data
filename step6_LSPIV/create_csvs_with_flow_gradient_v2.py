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

    # Load the data from your CSV file
    data = pd.read_csv(csv_file+'_no_outliers.csv', header=None, names=['X', 'Y', 'speed_X', 'speed_Y'])

    # Add a column for the speed magnitude for easier computation later
    data['speed'] = np.sqrt(data['speed_X']**2 + data['speed_Y']**2)

    # Create a KDTree for fast nearest-neighbor lookup
    coordinates = data[['X', 'Y']].values
    tree = KDTree(coordinates)

    # Function to calculate the gradient perpendicular to the flow
    def perpendicular_speed(data, index, tree):
        # Get the current speed components
        current_x, current_y = data.iloc[index]['speed_X'], data.iloc[index]['speed_Y']
        
        # Normalize the flow vector
        norm = np.sqrt(current_x**2 + current_y**2)
        if norm == 0:
            return np.nan  # No flow, return NaN

        # Calculate the 90 degrees left and right direction vectors
        left_direction = (-current_y / norm, current_x / norm)
        right_direction = (current_y / norm, -current_x / norm)

        value = False
        # Function to find the nearest neighbor in a specific direction
        def find_nearest_in_direction(direction):
            current_coords = data.iloc[index][['X', 'Y']].values
            # Move a small step in the direction to start searching
            search_coords = current_coords + np.array(direction)
            # Query the KDTree to find the closest point
            distance, nearest_index = tree.query(search_coords)
            return nearest_index if distance > 0 else None

        # Find left and right neighbors
        left_index = find_nearest_in_direction(left_direction)
        right_index = find_nearest_in_direction(right_direction)
        
        if left_index is not None and right_index is not None:
            # Get the speeds of the left and right cells
            left_speed = data.iloc[left_index]['speed']
            right_speed = data.iloc[right_index]['speed']
            current_speed = data.iloc[index]['speed']

            all_speeds = [current_speed,left_speed,right_speed]
            all_speeds.sort()
            # Calculate the gradient
            gradient = (abs(current_speed - left_speed) + abs(current_speed - right_speed)) #* all_speeds[0]
            if 6 < current_speed:
                print(current_speed,left_speed,right_speed)
                print(abs(current_speed - left_speed))
                print(abs(current_speed - right_speed))
                print(all_speeds[0])
                print('')
            return gradient
        
        return np.nan

    # Apply the perpendicular_speed function to every cell
    data['gradient'] = data.index.map(lambda i: perpendicular_speed(data, i, tree))

    # Save the results to a new CSV file
    data[['X', 'Y', 'gradient']].to_csv(csv_file+'_gradients.csv', index=False)
    data[['X', 'Y', 'speed']].to_csv(csv_file+'_speeds.csv', index=False)

