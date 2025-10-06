
from scipy.spatial import KDTree
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # to safely evaluate the string as a list
import random
import matplotlib.image as mpimg

# Scaling factors derived from your input
scaling_factor_x = 38.8  # X-axis scale from meters to pixels
scaling_factor_y = 38.8  # Y-axis scale from meters to pixels
x_pixel_offset = 1025  # Corresponds to real-world 25m in X
y_pixel_offset = 3536  # Corresponds to real-world 20m in Y

def real_world_to_pixel_x(x):
    """ Convert real-world X coordinate to pixel X coordinate. """
    return x_pixel_offset + scaling_factor_x * (x - 25)

def real_world_to_pixel_y(y):
    """ Convert real-world Y coordinate to pixel Y coordinate. """
    return y_pixel_offset - scaling_factor_y * (y - 20)

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

def plot_log_orientation_on_image(row, img, orientation):
    """ Plot the log orientation on the given image. """
    plt.imshow(img)
    
    if orientation == 'tlbr':
        top_left = parse_coordinates(row['top_left_geo'])
        bottom_right = parse_coordinates(row['bottom_right_geo'])
        if top_left is not None and bottom_right is not None:
            # Convert real-world coordinates to pixel coordinates
            pixel_top_left = [real_world_to_pixel_x(top_left[0]), real_world_to_pixel_y(top_left[1])]
            pixel_bottom_right = [real_world_to_pixel_x(bottom_right[0]), real_world_to_pixel_y(bottom_right[1])]
            
            # Plot the log line
            plt.plot([pixel_top_left[0], pixel_bottom_right[0]], [pixel_top_left[1], pixel_bottom_right[1]], 'r-', lw=3)
            plt.text(pixel_top_left[0], pixel_top_left[1], f'{top_left}', color='red')
            plt.text(pixel_bottom_right[0], pixel_bottom_right[1], f'{bottom_right}', color='red')
    else:
        top_right = parse_coordinates(row['top_right_geo'])
        bottom_left = parse_coordinates(row['bottom_left_geo'])
        if top_right is not None and bottom_left is not None:
            # Convert real-world coordinates to pixel coordinates
            pixel_top_right = [real_world_to_pixel_x(top_right[0]), real_world_to_pixel_y(top_right[1])]
            pixel_bottom_left = [real_world_to_pixel_x(bottom_left[0]), real_world_to_pixel_y(bottom_left[1])]
            
            # Plot the log line
            plt.plot([pixel_top_right[0], pixel_bottom_left[0]], [pixel_top_right[1], pixel_bottom_left[1]], 'r-', lw=3)
            plt.text(pixel_top_right[0], pixel_top_right[1], f'{top_right}', color='red')
            plt.text(pixel_bottom_left[0], pixel_bottom_left[1], f'{bottom_left}', color='red')

    # Display the calculated angle
    log_angle = row['log_angle']
    plt.text(50, 50, f'Detected Angle: {log_angle:.2f} degrees', fontsize=12, color='red')
    
    plt.title(f'Log ID: {row["ID"]}, Orientation: {orientation}')
    plt.show()

# Load the image
img = mpimg.imread('/media/jean-pierre/PortableSSD/15okt/velocities_median.jpg')

# Load the data
vidnumbers = [1, 2, 3, 4, 5]
total_velocity_df = pd.DataFrame()

for vidnumber in vidnumbers:
    # Load the data for each video
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data)
    
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

        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 1:
            velocity_log = np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x']) ** 2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y']) ** 2)

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
                'orientation': current_row['orientation'],
                'top_left_geo': current_row['top_left_geo'],
                'bottom_right_geo': current_row['bottom_right_geo'],
                'top_right_geo': current_row['top_right_geo'],
                'bottom_left_geo': current_row['bottom_left_geo'],
                'log_angle': log_angle,
                'velocity_log': velocity_log,
                'flow_speed': flow_speed,
                'normalized_velocity': normalized_velocity
            })

    print(f"Processed video {vidnumber}. Skipped rows due to missing data: {skipped_rows}")
    
    # Create a new dataframe from the collected velocity data
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

# Final dataframe size
print(f"Total data points: {len(total_velocity_df)}")

# Randomly select 5 rows to check
random_indices = random.sample(range(len(total_velocity_df)), 10)  # Pick 5 random rows
selected_rows = total_velocity_df.iloc[random_indices]

# Iterate over the selected rows and plot the orientations
for i, row in selected_rows.iterrows():
    print(f"Selected Row {i}:")
    print(row)
    
    # Plot the image with the log orientation and coordinates
    orientation = row['orientation']
    plot_log_orientation_on_image(row, img, orientation)

