import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon
import ast  # to safely evaluate the string as a list

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
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data)
    df['center_geo_x'] = pd.to_numeric(df['center_geo_x'], errors='coerce')
    df['center_geo_y'] = pd.to_numeric(df['center_geo_y'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.sort_values(by=['ID', 'timestamp'])

    flow_velocity_df = pd.read_csv('step6_LSPIV/all_vid' + str(vidnumber) + '_05_no_outliers.csv', 
                                   names=['X', 'Y', 'speed_x', 'speed_y'])
    flow_velocity_df['X'] = pd.to_numeric(flow_velocity_df['X'], errors='coerce')
    flow_velocity_df['Y'] = pd.to_numeric(flow_velocity_df['Y'], errors='coerce')
    flow_velocity_df['speed_x'] = pd.to_numeric(flow_velocity_df['speed_x'], errors='coerce')
    flow_velocity_df['speed_y'] = pd.to_numeric(flow_velocity_df['speed_y'], errors='coerce')

    velocity_coords = flow_velocity_df[['X', 'Y']].values
    kdtree = KDTree(velocity_coords)

    velocity_data = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.3:
            velocity_log = ( np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x']) ** 2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y']) ** 2) ) / 0.25

            mid_x = (current_row['center_geo_x'] + next_row['center_geo_x']) / 2
            mid_y = (current_row['center_geo_y'] + next_row['center_geo_y']) / 2

            _, idx = kdtree.query([mid_x, mid_y])
            flow_vector = flow_velocity_df.iloc[idx][['speed_x', 'speed_y']].values

            if current_row['orientation'] == 'tlbr':
                top_left = parse_coordinates(current_row['top_left_geo'])
                bottom_right = parse_coordinates(current_row['bottom_right_geo'])
                if top_left is not None and bottom_right is not None:
                    log_vector = bottom_right - top_left
                    log_angle = calculate_angle(log_vector, flow_vector)
                    print(log_angle)
                    flow_speed = np.linalg.norm(flow_vector)
                    normalized_velocity = velocity_log - flow_speed
                    velocity_data.append({
                        'vid_number': vidnumber,
                        'log_angle': log_angle,
                        'center_geo': current_row['center_geo']
                    })

    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

polygon1_coords = [(29.94, 65.35), (61.5, 29.8), (120.56, 56.76), (117.1, 81.54), (90.83, 90.06)]
polygon2_coords = [(212.49, 84.01), (214.51, 63.81), (189.87, 53.9), (167.66, 65.22), (157.06, 80.97), (163.82, 89.5)]
polygon1 = Polygon(polygon1_coords)
polygon2 = Polygon(polygon2_coords)

def classify_area(center_geo):
    xcoor = float((center_geo.split(',')[0]).split('[')[-1])
    ycoor = float((center_geo.split(',')[1]).split(']')[0])
    point = Point((xcoor, ycoor))
    if polygon1.contains(point) or polygon2.contains(point):
        return "diverging"
    else:
        return "converging"

total_velocity_df['area_pattern'] = total_velocity_df['center_geo'].apply(classify_area)

# KDE plot for diverging and converging
diverging_data = total_velocity_df[total_velocity_df['area_pattern'] == 'diverging']['log_angle']
converging_data = total_velocity_df[total_velocity_df['area_pattern'] == 'converging']['log_angle']

# Compute KDE for diverging data
diverging_kde = gaussian_kde(diverging_data)
diverging_x = np.linspace(min(diverging_data), max(diverging_data), 1000)
diverging_y = diverging_kde(diverging_x)

# Compute KDE for converging data
converging_kde = gaussian_kde(converging_data)
converging_x = np.linspace(min(converging_data), max(converging_data), 1000)
converging_y = converging_kde(converging_x)

# Plot KDEs
plt.figure(figsize=(6, 4))
plt.plot(diverging_x, diverging_y, label="Diverging", alpha=0.6)
plt.fill_between(diverging_x, diverging_y, alpha=0.3, label=None)
plt.plot(converging_x, converging_y, label="Converging", alpha=0.6)
plt.fill_between(converging_x, converging_y, alpha=0.3, label=None)

plt.title('Distribution of Log Angles with Respect to Flow Direction')
plt.xlabel('Log Angle (degrees)')
plt.ylabel('Density')
plt.grid(True)
plt.legend(title="Area Flow Pattern")
plt.tight_layout()  # Automatically adjusts subplot parameters to give some padding
plt.savefig('step10_analyze_rotation/log_angle_distribution_kde_diverging_converging.png')
plt.savefig('step10_analyze_rotation/log_angle_distribution_kde_diverging_converging.pdf')
plt.show()
