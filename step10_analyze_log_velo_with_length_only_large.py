import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import scipy.stats as stats
import matplotlib.pyplot as plt

vidnumbers = [1, 2, 3, 4, 5]

total_velocity_df = pd.DataFrame()

for vidnumber in vidnumbers:
    # Load the data
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
    
    df = pd.DataFrame(data)

    # Ensure columns 'center_geo_x', 'center_geo_y' and 'timestamp' are numeric
    df['center_geo_x'] = pd.to_numeric(df['center_geo_x'], errors='coerce')
    df['center_geo_y'] = pd.to_numeric(df['center_geo_y'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # Sort the dataframe by 'ID' and 'timestamp'
    df = df.sort_values(by=['ID', 'timestamp'])

    # Load the flow velocity data
    flow_velocity_df = pd.read_csv('step6_LSPIV/all_vid' + str(vidnumber) + '_05_speeds.csv', names=['X', 'Y', 'velocity'])

    # Convert X, Y, and velocity columns to numeric
    flow_velocity_df['X'] = pd.to_numeric(flow_velocity_df['X'], errors='coerce')
    flow_velocity_df['Y'] = pd.to_numeric(flow_velocity_df['Y'], errors='coerce')
    flow_velocity_df['velocity'] = pd.to_numeric(flow_velocity_df['velocity'], errors='coerce')

    # Build a KDTree for fast nearest-neighbor search
    velocity_coords = flow_velocity_df[['X', 'Y']].values
    kdtree = KDTree(velocity_coords)

    velocity_data = []

    # Iterate over the dataframe to find pairs of rows with the same 'ID' and consecutive timestamps
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.3:
            # Calculate log velocity
            velocity_log = (np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y'])**2)) / 0.25

            # Calculate midpoint
            mid_x = (current_row['center_geo_x'] + next_row['center_geo_x']) / 2
            mid_y = (current_row['center_geo_y'] + next_row['center_geo_y']) / 2

            # Find the closest flow velocity using KDTree
            _, idx = kdtree.query([mid_x, mid_y])
            closest_flow_velocity = float(flow_velocity_df.iloc[idx]['velocity'])  # Ensure it's a float

            # Append the result to the list
            velocity_data.append({
                'vid_number': vidnumber,
                'ID': current_row['ID'],
                'length_median': current_row['length_median'],
                'velocity_log': velocity_log,
                'closest_flow_velocity': closest_flow_velocity
            })

    # Create a new dataframe from the collected velocity data
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

print(total_velocity_df)

total_velocity_df = total_velocity_df[total_velocity_df['length_median'] >= 1]

# Now, analyze the correlation between velocity_log and length_median
total_velocity_df = total_velocity_df.dropna(subset=['velocity_log', 'length_median'])

print(total_velocity_df)

# Pearson correlation
pearson_corr, pearson_p_value = stats.pearsonr(total_velocity_df['length_median'], total_velocity_df['velocity_log'])
print(f"Pearson correlation coefficient: {pearson_corr:.3f}, P-value: {pearson_p_value:.3f}")

# Spearman correlation
spearman_corr, spearman_p_value = stats.spearmanr(total_velocity_df['length_median'], total_velocity_df['velocity_log'])
print(f"Spearman correlation coefficient: {spearman_corr:.3f}, P-value: {spearman_p_value:.3f}")

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(total_velocity_df['length_median'], total_velocity_df['velocity_log'])

# Plotting
plt.figure(figsize=(6, 4))
plt.scatter(total_velocity_df['length_median'], total_velocity_df['velocity_log'], alpha=0.6)
x_vals = np.array(plt.gca().get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red', label=f'Trend (Slope: {slope:.3f} \nP-value: {p_value:.3f})')
#plt.title('Length vs Log Velocity')
plt.xlabel('Length (m)')
plt.ylabel('Log Velocity (m/s)')
plt.ylim(0, 7)
plt.legend()
plt.tight_layout()
plt.savefig('step10_analyze_rotation/analyze_velocity_log_with_length_with_trendline_only_large.pdf')
plt.savefig('step10_analyze_rotation/analyze_velocity_log_with_length_with_trendline_only_large.png')
plt.show()
