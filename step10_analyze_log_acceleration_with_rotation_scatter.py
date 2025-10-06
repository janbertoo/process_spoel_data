import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
from scipy.spatial import KDTree
import ast  # For safely evaluating string representations of lists
import matplotlib.pyplot as plt

# Initialize variables
vidnumbers = [1, 2, 3, 4, 5]

# Define functions
def normalize_angle(angle):
    return angle % 360

def calculate_smallest_angle(top_left, bottom_right):
    dx = bottom_right[0] - top_left[0]
    dy = bottom_right[1] - top_left[1]
    angle = degrees(atan2(dy, dx))
    normalized_angle = normalize_angle(angle)
    if normalized_angle > 90:
        normalized_angle = 180 - normalized_angle
    return normalized_angle

def calculate_midpoint(point1, point2):
    mid_x = (point1[0] + point2[0]) / 2
    mid_y = (point1[1] + point2[1]) / 2
    return [mid_x, mid_y]

def parse_coordinates(coord):
    """Safely parse string representations of coordinates into Python lists."""
    try:
        return ast.literal_eval(coord) if isinstance(coord, str) else coord
    except (ValueError, SyntaxError):
        return None

# Initialize dataframes
total_rotation_df = pd.DataFrame()
total_velocity_df = pd.DataFrame()

# Process each video
for vidnumber in vidnumbers:
    with open(f'step9_dataframe/dataframe_vid{vidnumber}_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
    df = pd.DataFrame(data)
    
    # Parse and ensure numeric columns
    df['top_left_geo'] = df['top_left_geo'].apply(parse_coordinates)
    df['bottom_right_geo'] = df['bottom_right_geo'].apply(parse_coordinates)
    df['center_geo_x'] = pd.to_numeric(df['center_geo_x'], errors='coerce')
    df['center_geo_y'] = pd.to_numeric(df['center_geo_y'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    df = df.sort_values(by=['ID', 'timestamp'])

    # Load and clean flow velocity data
    gradient_df = pd.read_csv(f'step6_LSPIV/all_vid{vidnumber}_05_speeds.csv', names=['X', 'Y', 'gradient'])
    gradient_df = gradient_df.dropna(subset=['X', 'Y', 'gradient'])
    gradient_df['X'] = pd.to_numeric(gradient_df['X'], errors='coerce')
    gradient_df['Y'] = pd.to_numeric(gradient_df['Y'], errors='coerce')
    gradient_df['gradient'] = pd.to_numeric(gradient_df['gradient'], errors='coerce')
    gradient_df = gradient_df.dropna()  # Remove any remaining NaNs after conversion
    velocity_coords = gradient_df[['X', 'Y']].values
    kdtree = KDTree(velocity_coords)

    # Rotation computation
    rotation_data = []
    velocity_data = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) == 0.25:
            # Ensure the coordinates are parsed correctly
            if current_row['top_left_geo'] and current_row['bottom_right_geo']:
                angle1 = calculate_smallest_angle(current_row['top_left_geo'], current_row['bottom_right_geo'])
                angle2 = calculate_smallest_angle(next_row['top_left_geo'], next_row['bottom_right_geo'])
                rotation = abs(angle2 - angle1)
                rotation = min(rotation, 180 - rotation)  # Normalize to [0, 90]

                center_between_geo = calculate_midpoint([current_row['center_geo_x'], current_row['center_geo_y']],
                                                        [next_row['center_geo_x'], next_row['center_geo_y']])
                _, idx = kdtree.query(center_between_geo)
                closest_gradient = float(gradient_df.iloc[idx]['gradient'])
                rotation_data.append({
                    'vid_number': vidnumber,
                    'ID': current_row['ID'],
                    'timestamp': current_row['timestamp'],
                    'rotation': rotation,
                    'closest_gradient': closest_gradient
                })

            # Velocity computation
            velocity_log = (np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 +
                                    (next_row['center_geo_y'] - current_row['center_geo_y'])**2)) / 0.25
            velocity_data.append({
                'vid_number': vidnumber,
                'ID': current_row['ID'],
                'timestamp': current_row['timestamp'],
                'velocity_log': velocity_log
            })
    
    total_rotation_df = pd.concat([total_rotation_df, pd.DataFrame(rotation_data)], ignore_index=True)
    total_velocity_df = pd.concat([total_velocity_df, pd.DataFrame(velocity_data)], ignore_index=True)

# Merge the dataframes on ID and timestamp
merged_df = pd.merge(total_rotation_df, total_velocity_df, on=['vid_number', 'ID', 'timestamp'])

# Calculate acceleration
merged_df = merged_df.sort_values(by=['ID', 'timestamp'])
merged_df['acceleration'] = merged_df.groupby('ID')['velocity_log'].diff() / merged_df.groupby('ID')['timestamp'].diff()

# Filter rows with valid acceleration and rotation
merged_df = merged_df.dropna(subset=['acceleration', 'rotation'])

# Scatter plot: Acceleration vs Rotation
plt.figure(figsize=(6, 4))
plt.scatter(merged_df['acceleration'], merged_df['rotation'], alpha=0.6, c='purple')
plt.title('Acceleration vs. Rotation')
plt.xlabel('Acceleration (m/s²)')
plt.ylabel('Rotation (°)')
plt.tight_layout()
plt.savefig('acceleration_vs_rotation_scatter.pdf')
plt.savefig('acceleration_vs_rotation_scatter.png')
plt.show()

