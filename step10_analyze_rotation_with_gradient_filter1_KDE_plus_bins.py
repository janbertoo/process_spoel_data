import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
import ast
from scipy.spatial import KDTree  # For fast nearest-neighbor lookup
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize variables
vidnumbers = [1, 2, 3, 4, 5]
data = []
total_rotation_df = pd.DataFrame(data)

# Load and process data from each video number
for vidnumber in vidnumbers:
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    # Function to normalize the angle to the range [0, 360]
    def normalize_angle(angle):
        return angle % 360

    # Function to calculate the smallest angle between two logs
    def calculate_smallest_angle(top_left, bottom_right):
        dx = bottom_right[0] - top_left[0]
        dy = bottom_right[1] - top_left[1]
        angle = degrees(atan2(dy, dx))
        normalized_angle = normalize_angle(angle)
        
        if normalized_angle > 90:
            normalized_angle = 180 - normalized_angle
        
        return normalized_angle

    # Function to calculate the midpoint between two points
    def calculate_midpoint(point1, point2):
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        return [mid_x, mid_y]

    # Function to parse strings into lists of floats
    def parse_coordinates(coord_str):
        try:
            return ast.literal_eval(coord_str)
        except (ValueError, SyntaxError):
            return None

    def create_rotation_df(df, gradient_df, kdtree, gradient_coords):
        df['top_left_geo'] = df['top_left_geo'].apply(parse_coordinates)
        df['bottom_right_geo'] = df['bottom_right_geo'].apply(parse_coordinates)
        df['center_geo'] = df['center_geo'].apply(parse_coordinates)
        df['length'] = pd.to_numeric(df['length'], errors='coerce')
        df = df.sort_values(by=['ID', 'timestamp'])
        rotation_data = []

        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) == 0.25:
                if (current_row['top_left_geo'] and current_row['bottom_right_geo'] and
                    next_row['top_left_geo'] and next_row['bottom_right_geo']):
                    
                    if pd.notnull(current_row['length']) and pd.notnull(next_row['length']):
                        length_diff_percentage = abs(current_row['length'] - next_row['length']) / current_row['length']

                        if length_diff_percentage < 0.1:
                            angle1 = calculate_smallest_angle(current_row['top_left_geo'], current_row['bottom_right_geo'])
                            angle2 = calculate_smallest_angle(next_row['top_left_geo'], next_row['bottom_right_geo'])
                            rotation = abs(angle2 - angle1)

                            if rotation > 90:
                                rotation = 180 - rotation

                            abs_rotation = abs(rotation)

                            if current_row['center_geo'] and next_row['center_geo']:
                                center_between_geo = calculate_midpoint(current_row['center_geo'], next_row['center_geo'])
                                _, idx = kdtree.query(center_between_geo)
                                closest_gradient = gradient_df.iloc[idx]['gradient']

                                rotation_data.append({
                                    'vid_number': vidnumber,
                                    'ID': current_row['ID'],
                                    'length_median': current_row['length_median'],
                                    'diameter_median': current_row['diameter_median'],
                                    'volume_median': current_row['volume_median'],
                                    'timestamp_1': current_row['timestamp'],
                                    'timestamp_2': next_row['timestamp'],
                                    'rotation': rotation,
                                    'abs_rotation': abs_rotation,
                                    'center_between_geo': center_between_geo,
                                    'closest_gradient': closest_gradient
                                })

        rotation_df = pd.DataFrame(rotation_data)
        return rotation_df

    # Load your gradient CSV and ensure columns are numeric
    gradient_df = pd.read_csv('step6_LSPIV/all_vid'+str(vidnumber)+'_05_gradients.csv', names=['X', 'Y', 'gradient'])
    gradient_df['X'] = pd.to_numeric(gradient_df['X'], errors='coerce')
    gradient_df['Y'] = pd.to_numeric(gradient_df['Y'], errors='coerce')

    # Build a KDTree for fast nearest-neighbor search
    gradient_coords = gradient_df[['X', 'Y']].values
    kdtree = KDTree(gradient_coords)

    rotation_df = create_rotation_df(data, gradient_df, kdtree, gradient_coords)
    total_rotation_df = pd.concat([total_rotation_df, rotation_df])

# Data preparation and filtering
total_rotation_df = total_rotation_df[total_rotation_df['volume_median'] >= 0.00785398163397]
total_rotation_df = total_rotation_df[total_rotation_df['diameter_median'] >= 0.05]
total_rotation_df = total_rotation_df[total_rotation_df['length_median'] >= 1]

# Convert closest_gradient to numeric
total_rotation_df['closest_gradient'] = pd.to_numeric(total_rotation_df['closest_gradient'], errors='coerce')

# Create gradient bins of 0.25
bins = np.arange(0, total_rotation_df['closest_gradient'].max() + 0.25, 0.25)
labels = [f"{i}-{i + 0.25}" for i in bins[:-1]]
total_rotation_df['gradient_bins'] = pd.cut(total_rotation_df['closest_gradient'], bins=bins, labels=labels, right=False)

# Calculate averages per bin
average_rotations = total_rotation_df.groupby('gradient_bins')['abs_rotation'].mean().reset_index()

# Create a KDE plot
plt.figure(figsize=(5, 3))
sns.kdeplot(x=total_rotation_df['closest_gradient'], y=total_rotation_df['abs_rotation'], fill=True, cmap="Blues", thresh=0)

# Overlay the average line
x_values = average_rotations['gradient_bins'].cat.codes * 0.25 + 0.125  # Center of each bin for x-values
y_values = average_rotations['abs_rotation']  # Use the average rotation for y-values
plt.plot(x_values, y_values, color='red', linewidth=2, label='Average Rotation per Bin')

# Titles and labels
plt.title('KDE Plot of Gradient vs Rotation')
plt.xlabel('Gradient')
plt.ylabel('Rotation (° / 0.25 s)')
plt.xlim(0, 6)  # Set X-axis limit as needed
plt.ylim(0, 10)  # Set Y-axis limit as needed
plt.legend()
plt.tight_layout()

# Save and show plot
plt.savefig('step10_analyze_rotation/kde_plot_gradient_vs_rotation.png')
plt.savefig('step10_analyze_rotation/kde_plot_gradient_vs_rotation.pdf')
plt.show()
