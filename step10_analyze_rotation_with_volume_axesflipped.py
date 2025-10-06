import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
import ast
from scipy.spatial import KDTree
import scipy.stats as stats
import matplotlib.pyplot as plt

vidnumbers = [1, 2, 3, 4, 5]
data = []
total_rotation_df = pd.DataFrame(data)

for vidnumber in vidnumbers:
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

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
                if current_row['top_left_geo'] and current_row['bottom_right_geo'] and next_row['top_left_geo'] and next_row['bottom_right_geo']:
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

    gradient_df = pd.read_csv('step6_LSPIV/all_vid'+str(vidnumber)+'_05_speeds.csv', names=['X', 'Y', 'gradient'])
    gradient_df['X'] = pd.to_numeric(gradient_df['X'], errors='coerce')
    gradient_df['Y'] = pd.to_numeric(gradient_df['Y'], errors='coerce')
    gradient_coords = gradient_df[['X', 'Y']].values
    kdtree = KDTree(gradient_coords)

    rotation_df = create_rotation_df(data, gradient_df, kdtree, gradient_coords)
    total_rotation_df = pd.concat([total_rotation_df, rotation_df])

total_rotation_df = total_rotation_df[total_rotation_df['rotation'] <= 40]
total_rotation_df = total_rotation_df[total_rotation_df['volume_median'] >= 0.00785398163397]
total_rotation_df = total_rotation_df[total_rotation_df['diameter_median'] >= 0.1]
total_rotation_df = total_rotation_df[total_rotation_df['length_median'] >= 1]

var1 = total_rotation_df['volume_median']
var2 = total_rotation_df['abs_rotation']

var1 = pd.to_numeric(var1, errors='coerce')
var2 = pd.to_numeric(var2, errors='coerce')

pearson_corr, pearson_p_value = stats.pearsonr(var1, var2)
spearman_corr, spearman_p_value = stats.spearmanr(var1, var2)
slope, intercept, r_value, p_value, std_err = stats.linregress(var1, var2)

plt.figure(figsize=(4, 4))
plt.scatter(var1, var2, label='Data points', alpha=0.6)
x_vals = np.array(plt.gca().get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red', label=f'Trend (Slope: {slope:.3f} \nP-value: {p_value:.2f})')

#plt.title('Scatter plot of Volume vs Rotation')
plt.xlabel('Volume (m³)')
plt.ylabel('Rotation (° / 0.25 s)')
plt.legend()
plt.tight_layout()
plt.savefig('step10_analyze_rotation/analyze_volume_vs_rotation_with_trendline.png')
plt.savefig('step10_analyze_rotation/analyze_volume_vs_rotation_with_trendline.pdf')

