import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
import ast
from scipy.spatial import KDTree  # For fast nearest-neighbor lookup
import scipy.stats as stats
import matplotlib.pyplot as plt

vidnumbers = [1, 2, 3, 4, 5]

data = []
total_rotation_df = pd.DataFrame(data)

for vidnumber in vidnumbers:
    # Replace 'your_file.pkl' with the path to your pickle file
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
        
        # Ensure the smallest angle between two logs, restrict to [0, 90] degrees
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

    # Assuming your dataframe is called df and the gradient CSV is loaded as gradient_df
    def create_rotation_df(df, gradient_df, kdtree, gradient_coords):
        # Parse the coordinate columns to ensure they are lists
        df['top_left_geo'] = df['top_left_geo'].apply(parse_coordinates)
        df['bottom_right_geo'] = df['bottom_right_geo'].apply(parse_coordinates)
        df['center_geo'] = df['center_geo'].apply(parse_coordinates)

        # Convert the 'length' column to numeric values (floats)
        df['length'] = pd.to_numeric(df['length'], errors='coerce')

        # Sort the dataframe by 'ID' and 'timestamp'
        df = df.sort_values(by=['ID', 'timestamp'])

        # Create a list to store results
        rotation_data = []

        # Iterate over the dataframe to find pairs of rows with the same 'ID' and timestamp difference of 0.25
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # Check if the ID matches and timestamp difference is 0.25
            if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) == 0.25:
                # Ensure the coordinates are parsed correctly
                if current_row['top_left_geo'] and current_row['bottom_right_geo'] and next_row['top_left_geo'] and next_row['bottom_right_geo']:
                    # Calculate the length difference as a percentage
                    if pd.notnull(current_row['length']) and pd.notnull(next_row['length']):
                        length_diff_percentage = abs(current_row['length'] - next_row['length']) / current_row['length']

                        # Only proceed if the length difference is less than 10%
                        if length_diff_percentage < 0.1:
                            # Calculate the smallest rotation angle between the two rows
                            angle1 = calculate_smallest_angle(current_row['top_left_geo'], current_row['bottom_right_geo'])
                            angle2 = calculate_smallest_angle(next_row['top_left_geo'], next_row['bottom_right_geo'])
                            rotation = abs(angle2 - angle1)

                            # Ensure the rotation angle does not exceed 90 degrees
                            if rotation > 90:
                                rotation = 180 - rotation

                            abs_rotation = abs(rotation)

                            # Calculate the midpoint of the 'center_geo' coordinates
                            if current_row['center_geo'] and next_row['center_geo']:
                                center_between_geo = calculate_midpoint(current_row['center_geo'], next_row['center_geo'])

                                # Find the closest gradient using KDTree
                                _, idx = kdtree.query(center_between_geo)
                                closest_gradient = gradient_df.iloc[idx]['gradient']

                                # Append the result to the list
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

        # Create a new dataframe from the collected rotation data
        rotation_df = pd.DataFrame(rotation_data)
        return rotation_df

    # Load your gradient CSV and ensure columns are numeric
    gradient_df = pd.read_csv('step6_LSPIV/all_vid'+str(vidnumber)+'_05_gradients.csv', names=['X', 'Y', 'gradient'])

    # Convert X and Y columns to numeric
    gradient_df['X'] = pd.to_numeric(gradient_df['X'], errors='coerce')
    gradient_df['Y'] = pd.to_numeric(gradient_df['Y'], errors='coerce')

    # Build a KDTree for fast nearest-neighbor search
    gradient_coords = gradient_df[['X', 'Y']].values
    kdtree = KDTree(gradient_coords)

    # Example usage with your dataframe df
    rotation_df = create_rotation_df(data, gradient_df, kdtree, gradient_coords)

    # Concatenate results for all video numbers
    total_rotation_df = pd.concat([total_rotation_df, rotation_df])

print(total_rotation_df)

# Filter rows where 'volume_median' is greater than or equal to 0.00785398163397
total_rotation_df = total_rotation_df[total_rotation_df['volume_median'] >= 0.00785398163397]
total_rotation_df = total_rotation_df[total_rotation_df['diameter_median'] >= 0.05]
total_rotation_df = total_rotation_df[total_rotation_df['length_median'] >= 1]
'''
# Calculate the interquartile range (IQR) for outlier detection
Q1 = total_rotation_df['abs_rotation'].quantile(0.25)
Q3 = total_rotation_df['abs_rotation'].quantile(0.75)
IQR = Q3 - Q1

multiplier = 5
# Define outlier bounds
lower_bound = Q1 - multiplier * IQR
upper_bound = Q3 + multiplier    * IQR

# Filter the DataFrame to remove outliers
total_rotation_df = total_rotation_df[(total_rotation_df['abs_rotation'] >= lower_bound) & (total_rotation_df['abs_rotation'] <= upper_bound)]
'''

# Example for analyzing correlation between 'var1' and 'var2'
var1 = total_rotation_df['abs_rotation']
var2 = total_rotation_df['closest_gradient']

# Convert variables to numeric, forcing errors to NaN
var1 = pd.to_numeric(var1, errors='coerce')
var2 = pd.to_numeric(var2, errors='coerce')

# 1. Pearson correlation coefficient
pearson_corr, pearson_p_value = stats.pearsonr(var1, var2)
print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"P-value: {pearson_p_value:.3f}")

# 2. Spearman rank correlation
spearman_corr, spearman_p_value = stats.spearmanr(var1, var2)
print(f"Spearman correlation coefficient: {spearman_corr:.3f}")
print(f"P-value: {spearman_p_value:.3f}")

# Perform linear regression for trendline
slope, intercept, r_value, p_value, std_err = stats.linregress(var1, var2)

# Generate the trendline values
trendline = slope * var1 + intercept

plt.figure(figsize=(5, 3))  # Change the width and height (in inches) as needed

# Plotting the scatter plot with trendline
plt.scatter(var1, var2, label='Data points', alpha=0.6)
plt.plot(var1, trendline, color='red', label=f'Trend (Slope: {slope:.3f} \nP-value={p_value:.2f})')

# Add labels, title, and legend
plt.title('Scatter plot of Rotation vs Gradient')
plt.xlabel('Rotation (° / 0.25 s)')
plt.ylabel('Gradient')
plt.legend()
plt.tight_layout()

plt.savefig('step10_analyze_rotation/analyze_rotation_vs_gradient.pdf')
plt.savefig('step10_analyze_rotation/analyze_rotation_vs_gradient.png')

# Show the plot
plt.show()

# Print the p-value of the trendline significance
print(f"Trendline p-value: {p_value:.3f}")
print(f"Trendline slope: {slope:.3f}")
