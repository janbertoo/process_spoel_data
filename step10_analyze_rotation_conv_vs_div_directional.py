import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import ast  # For safely parsing string to list

# Function to classify areas
def classify_area(center_geo, polygon1, polygon2):
    xcoor = center_geo[0]
    ycoor = center_geo[1]
    point = Point((xcoor, ycoor))
    if polygon1.contains(point) or polygon2.contains(point):
        return "diverging"
    else:
        return "converging"

# Function to calculate directional rotation
def calculate_directional_rotation(top_left1, bottom_right1, top_left2, bottom_right2):
    def angle(top_left, bottom_right):
        dx = bottom_right[0] - top_left[0]
        dy = bottom_right[1] - top_left[1]
        return degrees(atan2(dy, dx))
    
    angle1 = angle(top_left1, bottom_right1)
    angle2 = angle(top_left2, bottom_right2)
    rotation = angle2 - angle1  # Keep the sign for direction
    rotation = (rotation + 180) % 360 - 180  # Normalize to [-180, 180]
    return rotation

# Load polygons
polygon1_coords = [(29.94, 65.35), (61.5, 29.8), (120.56, 56.76), (117.1, 81.54), (90.83, 90.06)]
polygon2_coords = [(212.49, 84.01), (214.51, 63.81), (189.87, 53.9), (167.66, 65.22), (157.06, 80.97), (163.82, 89.5)]
polygon1 = Polygon(polygon1_coords)
polygon2 = Polygon(polygon2_coords)

# Process data
vidnumbers = [1, 2, 3, 4, 5]
rotation_data = []

for vidnumber in vidnumbers:
    with open(f'step9_dataframe/dataframe_vid{vidnumber}_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
    df = pd.DataFrame(data)

    df['top_left_geo'] = df['top_left_geo'].apply(ast.literal_eval)
    df['bottom_right_geo'] = df['bottom_right_geo'].apply(ast.literal_eval)
    df['center_geo'] = df['center_geo'].apply(ast.literal_eval)
    df = df.sort_values(by=['ID', 'timestamp'])

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.25:
            directional_rotation = calculate_directional_rotation(
                current_row['top_left_geo'], current_row['bottom_right_geo'],
                next_row['top_left_geo'], next_row['bottom_right_geo']
            )
            center_between = [(current_row['center_geo'][0] + next_row['center_geo'][0]) / 2,
                              (current_row['center_geo'][1] + next_row['center_geo'][1]) / 2]
            area = classify_area(center_between, polygon1, polygon2)
            rotation_data.append({'directional_rotation': directional_rotation, 'area': area})

rotation_df = pd.DataFrame(rotation_data)

# KDE plots for directional rotation
diverging_data = rotation_df[rotation_df['area'] == 'diverging']['directional_rotation']
converging_data = rotation_df[rotation_df['area'] == 'converging']['directional_rotation']

diverging_kde = gaussian_kde(diverging_data)
converging_kde = gaussian_kde(converging_data)

x_range = np.linspace(-90, 90, 1000)
diverging_y = diverging_kde(x_range)
converging_y = converging_kde(x_range)

plt.figure(figsize=(6, 4))
plt.plot(x_range, diverging_y, label='Diverging', alpha=0.6)
plt.plot(x_range, converging_y, label='Converging', alpha=0.6)
plt.fill_between(x_range, diverging_y, alpha=0.3)
plt.fill_between(x_range, converging_y, alpha=0.3)
plt.xlabel('Directional Rotation (degrees / 0.25s)')
plt.ylabel('Density')
plt.title('Directional Rotation KDE in Converging and Diverging Areas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('directional_rotation_kde_diverging_converging.png')
plt.show()

