import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
from scipy.spatial import KDTree
import ast  # For safely evaluating string representations of lists
import seaborn as sns
import matplotlib.pyplot as plt

# Define grid resolution
grid_size = 0.5  # 0.5 meters

# Initialize variables
vidnumbers = [1, 2, 3, 4, 5]

# Define helper functions
def parse_coordinates(coord):
    """Safely parse string representations of coordinates into Python lists."""
    try:
        return ast.literal_eval(coord) if isinstance(coord, str) else coord
    except (ValueError, SyntaxError):
        return None

# Initialize dataframes
total_velocity_df = pd.DataFrame()

# Process each video
for vidnumber in vidnumbers:
    with open(f'step9_dataframe/dataframe_vid{vidnumber}_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
    df = pd.DataFrame(data)
    
    # Parse and ensure numeric columns
    df['center_geo_x'] = pd.to_numeric(df['center_geo_x'], errors='coerce')
    df['center_geo_y'] = pd.to_numeric(df['center_geo_y'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    df = df.sort_values(by=['ID', 'timestamp'])

    # Compute velocity
    velocity_data = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) == 0.25:
            velocity_log = (np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 +
                                    (next_row['center_geo_y'] - current_row['center_geo_y'])**2)) / 0.25
            velocity_data.append({
                'vid_number': vidnumber,
                'ID': current_row['ID'],
                'timestamp': current_row['timestamp'],
                'center_geo_x': current_row['center_geo_x'],
                'center_geo_y': current_row['center_geo_y'],
                'velocity_log': velocity_log
            })
    
    total_velocity_df = pd.concat([total_velocity_df, pd.DataFrame(velocity_data)], ignore_index=True)

# Filter rows with valid velocity
total_velocity_df = total_velocity_df.dropna(subset=['velocity_log'])

# Create a grid
x_min, x_max = total_velocity_df['center_geo_x'].min(), total_velocity_df['center_geo_x'].max()
y_min, y_max = total_velocity_df['center_geo_y'].min(), total_velocity_df['center_geo_y'].max()

x_bins = np.arange(x_min, x_max + grid_size, grid_size)
y_bins = np.arange(y_min, y_max + grid_size, grid_size)

# Bin data into the grid and compute average log velocity
grid = total_velocity_df.copy()
grid['x_bin'] = pd.cut(grid['center_geo_x'], bins=x_bins, labels=x_bins[:-1])
grid['y_bin'] = pd.cut(grid['center_geo_y'], bins=y_bins, labels=y_bins[:-1])
grid = grid.dropna(subset=['x_bin', 'y_bin'])

# Group by grid cells and calculate average log velocity
average_velocity = grid.groupby(['x_bin', 'y_bin'])['velocity_log'].mean().reset_index()
average_velocity['x_bin'] = average_velocity['x_bin'].astype(float)
average_velocity['y_bin'] = average_velocity['y_bin'].astype(float)

# Plot the grid with average log velocity
plt.figure(figsize=(13, 5))
scatter = plt.scatter(
    average_velocity['x_bin'],
    average_velocity['y_bin'],
    c=average_velocity['velocity_log'],
    cmap='RdYlGn',
    s=4,
    vmin=0,  # Set the minimum of the color scale
    vmax=5    # Set the maximum of the color scale
)
plt.colorbar(scatter, label='Average Log Velocity (m/s)')
plt.title('Spatial Distribution of Average Log Velocity')
plt.axis('equal')
plt.xlabel('Center Geo X (m)')
plt.ylabel('Center Geo Y (m)')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save and show the plot
plt.savefig('step10_analyze_rotation/spatial_log_velocity_grid.pdf')
plt.savefig('step10_analyze_rotation/spatial_log_velocity_grid.png')
plt.show()
