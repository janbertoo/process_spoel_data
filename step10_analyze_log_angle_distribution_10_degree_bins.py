import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import ast  # to safely evaluate the string as a list
from shapely.geometry import Point, Polygon

def parse_coordinates(coord_str):
    """ Safely parse the coordinate string to a list of floats. """
    try:
        return np.array(ast.literal_eval(coord_str))
    except (ValueError, SyntaxError):
        return None  # Handle invalid or missing data

def calculate_angle(v1, v2):
    """ 
    Calculate the smallest angle between two vectors v1 and v2.
    The result is in degrees and cannot exceed 90 degrees by definition.
    """
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(min(angle, np.pi - angle))

vidnumbers = [1, 2, 3, 4, 5]
total_velocity_df = pd.DataFrame()

for vidnumber in vidnumbers:
    # Load the data for each video
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data)
    print(df)
    
    # Ensure columns 'center_geo_x', 'center_geo_y', and 'timestamp' are numeric
    df['center_geo_x'] = pd.to_numeric(df['center_geo_x'], errors='coerce')
    df['center_geo_y'] = pd.to_numeric(df['center_geo_y'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # Sort the dataframe by 'ID' and 'timestamp'
    df = df.sort_values(by=['ID', 'timestamp'])
    print(f"Loaded data for video {vidnumber}. Rows: {len(df)}")

    # Load the flow direction data
    flow_velocity_df = pd.read_csv('step6_LSPIV/all_vid' + str(vidnumber) + '_05_no_outliers.csv', names=['X', 'Y', 'speed_x', 'speed_y'])
    
    # Convert X, Y, and velocity columns to numeric
    flow_velocity_df['X'] = pd.to_numeric(flow_velocity_df['X'], errors='coerce')
    flow_velocity_df['Y'] = pd.to_numeric(flow_velocity_df['Y'], errors='coerce')
    flow_velocity_df['speed_x'] = pd.to_numeric(flow_velocity_df['speed_x'], errors='coerce')
    flow_velocity_df['speed_y'] = pd.to_numeric(flow_velocity_df['speed_y'], errors='coerce')

    # Build a KDTree for fast nearest-neighbor search
    velocity_coords = flow_velocity_df[['X', 'Y']].values
    kdtree = KDTree(velocity_coords)

    velocity_data = []
    skipped_rows = 0

    # Iterate over the dataframe to calculate log orientation and flow angle
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.3:
            velocity_log = ( np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x']) ** 2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y']) ** 2) ) / 0.25

            # Find the midpoint
            mid_x = (current_row['center_geo_x'] + next_row['center_geo_x']) / 2
            mid_y = (current_row['center_geo_y'] + next_row['center_geo_y']) / 2

            # Find the closest flow velocity using KDTree
            _, idx = kdtree.query([mid_x, mid_y])
            flow_vector = flow_velocity_df.iloc[idx][['speed_x', 'speed_y']].values

            # Parse the log corner coordinates
            if current_row['orientation'] == 'tlbr':
                top_left = parse_coordinates(current_row['top_left_geo'])
                bottom_right = parse_coordinates(current_row['bottom_right_geo'])
                if top_left is not None and bottom_right is not None:
                    log_vector = bottom_right - top_left
                else:
                    skipped_rows += 1
                    continue  # Skip this row if the coordinates cannot be parsed
            else:
                top_right = parse_coordinates(current_row['top_right_geo'])
                bottom_left = parse_coordinates(current_row['bottom_left_geo'])
                if top_right is not None and bottom_left is not None:
                    log_vector = bottom_left - top_right
                else:
                    skipped_rows += 1
                    continue  # Skip this row if the coordinates cannot be parsed

            # Calculate the smallest angle between log orientation and flow direction
            log_angle = calculate_angle(log_vector, flow_vector)

            # Calculate normalized velocity
            flow_speed = np.linalg.norm(flow_vector)
            normalized_velocity = velocity_log - flow_speed

            # Append the result to the list
            velocity_data.append({
                'vid_number': vidnumber,
                'ID': current_row['ID'],
                'log_angle': log_angle,
                'length_median': current_row['length_median'],
                'diameter_median': current_row['diameter_median'],
                'volume_median': current_row['volume_median'],
                'drone_number': current_row['drone_number'],
                'velocity_log': velocity_log,
                'flow_speed': flow_speed,
                'normalized_velocity': normalized_velocity,
            })

    print(f"Processed video {vidnumber}. Skipped rows due to missing data: {skipped_rows}")
    
    # Create a new dataframe from the collected velocity data
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

# Filter data
total_velocity_df = total_velocity_df[total_velocity_df['diameter_median'] >= 0.05]
total_velocity_df = total_velocity_df[total_velocity_df['length_median'] >= 1]

# Extract log angles for plotting
log_angles = total_velocity_df['log_angle']

# KDE plot
plt.figure(figsize=(8, 6))
sns.kdeplot(log_angles, fill=True, alpha=0.6, color="blue")
plt.title('Distribution of Log Angles with Respect to Flow Direction')
plt.xlabel('Log Angle (degrees)')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('log_angle_distribution_kde.png')
plt.show()



