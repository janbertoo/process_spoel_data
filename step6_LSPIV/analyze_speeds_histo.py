import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV files
file_paths = [
    'all_vid1_05.csv', 
    'all_vid2_05.csv', 
    'all_vid3_05.csv', 
    'all_vid4_05.csv', 
    'all_vid5_05.csv'
]

# Load all videos' speed data
dfs = []
for file in file_paths:
    df = pd.read_csv(file, header=None, names=['X', 'Y', 'speed_X', 'speed_Y'])
    # Calculate total velocity from X and Y components
    df['velocity'] = np.sqrt(df['speed_X']**2 + df['speed_Y']**2)
    # Filter out velocities less than 0.25 m/s
    df = df[df['velocity'] >= 0.25]
    dfs.append(df)

# Concatenate all velocities into a single array
all_velocities = np.concatenate([df['velocity'].values for df in dfs])

# Plot histogram
plt.figure(figsize=(5, 4))
plt.hist(all_velocities, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('LSPIV Velocity (m/s)')
plt.ylabel('Frequency')
#plt.title('Histogram of log velocities')
plt.xlim(0, 8)  # Limiting the x-axis
#plt.grid(True, linestyle='--', alpha=0.5)
# Calculate mean and median
mean_velocity = np.mean(all_velocities)
median_velocity = np.median(all_velocities)
# Add mean and median lines
plt.axvline(mean_velocity, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_velocity:.2f} m/s')
plt.axvline(median_velocity, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_velocity:.2f} m/s')
plt.legend()
# Save the figure
plt.savefig('velocity_histogram.pdf', bbox_inches='tight')

# Show the plot
plt.show()
