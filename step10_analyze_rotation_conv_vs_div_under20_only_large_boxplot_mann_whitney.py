import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde, mannwhitneyu
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import ast  # For safely parsing string to list
import seaborn as sns  # Optional for aesthetic boxplots
from scipy.stats import ks_2samp





# Function to classify areas
def classify_area(center_geo, polygon1, polygon2):
    xcoor = center_geo[0]
    ycoor = center_geo[1]
    point = Point((xcoor, ycoor))
    if polygon1.contains(point) or polygon2.contains(point):
        return "diverging"
    else:
        return "converging"

# Function to calculate rotation
def calculate_rotation(top_left1, bottom_right1, top_left2, bottom_right2):
    def angle(top_left, bottom_right):
        dx = bottom_right[0] - top_left[0]
        dy = bottom_right[1] - top_left[1]
        return degrees(atan2(dy, dx))
    
    angle1 = angle(top_left1, bottom_right1)
    angle2 = angle(top_left2, bottom_right2)
    rotation = abs(angle2 - angle1)
    if rotation > 180:
        rotation = 360 - rotation
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

    # Filter rows based on length_median and diameter_median
    df = df[(df['length_median'] > 1) & (df['diameter_median'] > 0.1)]

    df = df.sort_values(by=['ID', 'timestamp'])

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.25:
            rotation = calculate_rotation(
                current_row['top_left_geo'], current_row['bottom_right_geo'],
                next_row['top_left_geo'], next_row['bottom_right_geo']
            )
            # Discard datapoints with rotation > 20 degrees
            if rotation > 20:
                continue
            center_between = [(current_row['center_geo'][0] + next_row['center_geo'][0]) / 2,
                              (current_row['center_geo'][1] + next_row['center_geo'][1]) / 2]
            area = classify_area(center_between, polygon1, polygon2)
            rotation_data.append({'rotation': rotation, 'area': area})

rotation_df = pd.DataFrame(rotation_data)

# KDE plots
diverging_data = rotation_df[rotation_df['area'] == 'diverging']['rotation']
converging_data = rotation_df[rotation_df['area'] == 'converging']['rotation']

diverging_kde = gaussian_kde(diverging_data)
converging_kde = gaussian_kde(converging_data)

x_range = np.linspace(0, max(rotation_df['rotation']), 1000)
diverging_y = diverging_kde(x_range)
converging_y = converging_kde(x_range)

plt.figure(figsize=(6, 4))
plt.plot(x_range, diverging_y, label='Diverging', alpha=0.6)
plt.plot(x_range, converging_y, label='Converging', alpha=0.6)
plt.fill_between(x_range, diverging_y, alpha=0.3)
plt.fill_between(x_range, converging_y, alpha=0.3)
plt.xlabel('Wood Piece Rotation (° / 0.25 s)')
plt.ylabel('Density')
plt.xlim(0, 20)
plt.title('Rotation KDE in Converging and Diverging Areas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('step10_analyze_rotation/rotation_kde_diverging_converging_only_large.png')
plt.savefig('step10_analyze_rotation/rotation_kde_diverging_converging_only_large.pdf')
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=rotation_df, x='area', y='rotation')
plt.xlabel('Area Type')
plt.ylabel('Rotation (degrees / 0.25s)')
plt.title('Boxplot of Rotation in Converging and Diverging Areas')
plt.grid(True)
plt.tight_layout()
plt.savefig('step10_analyze_rotation/rotation_boxplot_diverging_converging_only_large.png')
plt.savefig('step10_analyze_rotation/rotation_boxplot_diverging_converging_only_large.pdf')
plt.show()

# Mann-Whitney U Test
stat, p_value = mannwhitneyu(diverging_data, converging_data, alternative='two-sided')

print(f"Mann-Whitney U Test Statistic: {stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("There is a statistically significant difference between the two groups.")
else:
    print("There is no statistically significant difference between the two groups.")

ks_stat, ks_p_value = ks_2samp(diverging_data, converging_data)

print(f"Kolmogorov-Smirnov Test Statistic: {ks_stat}")
print(f"P-Value: {ks_p_value}")

if ks_p_value < 0.05:
    print("There is a statistically significant difference in the distributions of the two groups.")
else:
    print("There is no statistically significant difference in the distributions of the two groups.")
