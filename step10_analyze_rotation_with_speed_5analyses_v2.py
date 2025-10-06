import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For KDE and violin plots
from scipy.spatial import KDTree
import ast
from math import atan2, degrees

# Load gradient data for KDTree lookup
gradient_df = pd.read_csv('step6_LSPIV/all_vid1_05_speeds.csv', names=['X', 'Y', 'gradient'])  # Update path if needed
gradient_df['X'] = pd.to_numeric(gradient_df['X'], errors='coerce')
gradient_df['Y'] = pd.to_numeric(gradient_df['Y'], errors='coerce')

# Build a KDTree for fast nearest-neighbor search
gradient_coords = gradient_df[['X', 'Y']].values
kdtree = KDTree(gradient_coords)

# Load your data
vidnumbers = [1, 2, 3, 4, 5]
total_rotation_df = pd.DataFrame()

# Function to normalize angle to [0, 90] degrees
def normalize_angle(angle):
    angle = angle % 180
    if angle > 90:
        angle = 180 - angle
    return angle

# Function to calculate rotation between two frames
def calculate_rotation(row1, row2):
    angle1 = normalize_angle(degrees(atan2(row1['bottom_right_geo'][1] - row1['top_left_geo'][1],
                                           row1['bottom_right_geo'][0] - row1['top_left_geo'][0])))
    angle2 = normalize_angle(degrees(atan2(row2['bottom_right_geo'][1] - row2['top_left_geo'][1],
                                           row2['bottom_right_geo'][0] - row2['top_left_geo'][0])))
    return abs(angle2 - angle1)

# Parse coordinates from strings to lists
def parse_coordinates(coord_str):
    try:
        return ast.literal_eval(coord_str)
    except (ValueError, SyntaxError):
        return None

# Calculate the closest gradient using KDTree
def get_closest_gradient(center_geo):
    _, idx = kdtree.query(center_geo)
    return gradient_df.iloc[idx]['gradient']

# Load data for each video number and calculate abs_rotation and closest_gradient
for vidnumber in vidnumbers:
    with open(f'step9_dataframe/dataframe_vid{vidnumber}_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
        
        # Ensure top_left_geo and bottom_right_geo columns are parsed correctly
        data['top_left_geo'] = data['top_left_geo'].apply(parse_coordinates)
        data['bottom_right_geo'] = data['bottom_right_geo'].apply(parse_coordinates)
        data['center_geo'] = data['center_geo'].apply(parse_coordinates)
        
        # Sort data by ID and timestamp for calculating rotation
        data = data.sort_values(by=['ID', 'timestamp'])
        data['abs_rotation'] = np.nan  # Initialize the column
        data['closest_gradient'] = np.nan  # Initialize the column

        # Calculate abs_rotation and closest_gradient between consecutive frames
        for i in range(len(data) - 1):
            if data.iloc[i]['ID'] == data.iloc[i + 1]['ID']:
                rotation = calculate_rotation(data.iloc[i], data.iloc[i + 1])
                data.at[data.index[i], 'abs_rotation'] = rotation

                # Calculate closest gradient for midpoint of center_geo coordinates
                if data.iloc[i]['center_geo'] and data.iloc[i + 1]['center_geo']:
                    center_between_geo = [(data.iloc[i]['center_geo'][0] + data.iloc[i + 1]['center_geo'][0]) / 2,
                                          (data.iloc[i]['center_geo'][1] + data.iloc[i + 1]['center_geo'][1]) / 2]
                    data.at[data.index[i], 'closest_gradient'] = get_closest_gradient(center_between_geo)

        total_rotation_df = pd.concat([total_rotation_df, data])

# Reset index to remove duplicate labels
total_rotation_df.reset_index(drop=True, inplace=True)

# Apply filters
total_rotation_df = total_rotation_df[total_rotation_df['volume_median'] >= 0.00785398163397]
total_rotation_df = total_rotation_df[total_rotation_df['diameter_median'] >= 0.05]
total_rotation_df = total_rotation_df[total_rotation_df['length_median'] >= 1]

# Variables for analysis
var1 = pd.to_numeric(total_rotation_df['abs_rotation'], errors='coerce')  # Rotation
var2 = pd.to_numeric(total_rotation_df['closest_gradient'], errors='coerce')  # Velocity

# Drop NaN values
var1 = var1.dropna()
var2 = var2.dropna()

# Plot 1: KDE Plot
plt.figure(figsize=(5, 3))
sns.kdeplot(x=var2, y=var1, fill=True, cmap="Blues")  # Swap axes
plt.title('KDE Plot of Velocity vs Rotation')
plt.xlabel('LSPI Velocity (m/s)')
plt.ylabel('Rotation (° / 0.25 s)')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/kde_velocity_vs_rotation.png')
plt.show()

# Plot 2: Hexbin Plot
plt.figure(figsize=(5, 3))
plt.hexbin(var2, var1, gridsize=30, cmap='Blues')  # Swap axes
plt.colorbar(label='Density')
plt.title('Hexbin Plot of Velocity vs Rotation')
plt.xlabel('LSPI Velocity (m/s)')
plt.ylabel('Rotation (° / 0.25 s)')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/hexbin_velocity_vs_rotation.png')
plt.show()

# Plot 3: 2D Histogram
plt.figure(figsize=(5, 3))
plt.hist2d(var2, var1, bins=(30, 30), cmap='Blues')  # Swap axes
plt.colorbar(label='Count')
plt.title('2D Histogram of Velocity vs Rotation')
plt.xlabel('LSPI Velocity (m/s)')
plt.ylabel('Rotation (° / 0.25 s)')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/hist2d_velocity_vs_rotation.png')
plt.show()

# Plot 4: Boxplot of Rotation by Velocity Range
velocity_bins = pd.cut(var2, bins=np.arange(var2.min(), var2.max() + 0.5, 0.5), right=False)  # Binning velocity into 0.5 m/s intervals
velocity_bins = pd.Categorical(velocity_bins)  # Ensure it's treated as categorical
plt.figure(figsize=(8, 4))
sns.boxplot(x=var1, y=velocity_bins, palette="Blues")  # Plot rotation against binned velocity
plt.xticks(rotation=45)
plt.title('Boxplot of Rotation by Velocity Range')
plt.xlabel('Rotation (°)')
plt.ylabel('Velocity Range (m/s)')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/boxplot_rotation_vs_velocity.png')
plt.show()

# Plot 5: Violin Plot of Velocity by Rotation Range
plt.figure(figsize=(8, 4))
sns.violinplot(x=var1, y=velocity_bins, palette="Blues")  # Plot rotation against binned velocity
plt.xticks(rotation=45)
plt.title('Violin Plot of Rotation by Velocity Range')
plt.xlabel('Rotation (°)')
plt.ylabel('Velocity Range (m/s)')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/violinplot_rotation_vs_velocity.png')
plt.show()

# Optional: Logarithmic Transformation
var1_log = np.log1p(var1)  # log(1 + x) to handle zero values
var2_log = np.log1p(var2)

plt.figure(figsize=(5, 3))
plt.scatter(var2_log, var1_log, alpha=0.6)  # Swap axes
plt.title('Scatter plot of Log(Velocity) vs Log(Rotation)')
plt.xlabel('Log(Velocity)')
plt.ylabel('Log(Rotation)')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/log_scatter_velocity_vs_rotation.png')
plt.show()
