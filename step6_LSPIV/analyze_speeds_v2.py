import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Load CSV files
file_paths = [
    'all_vid1_05.csv', 
    'all_vid2_05.csv', 
    'all_vid3_05.csv', 
    'all_vid4_05.csv', 
    'all_vid5_05.csv',  # These are the 5 video files
    'median_speeds.csv'  # This is for median velocities (we're not calculating average for this one)
]

# Load all videos' speed data
dfs = []
for file in file_paths[:-1]:  # Load only the first 5 video files, excluding the median file
    df = pd.read_csv(file, header=None, names=['X', 'Y', 'speed_X', 'speed_Y'])
    # Calculate total velocity from X and Y components
    df['velocity'] = np.sqrt(df['speed_X']**2 + df['speed_Y']**2)
    dfs.append(df)

# Load median speeds (as the 6th file)
median_df = pd.read_csv(file_paths[-1], header=None, names=['X', 'Y', 'speed_X', 'speed_Y'])
median_df['median_velocity'] = np.sqrt(median_df['speed_X']**2 + median_df['speed_Y']**2)

# Merge all video data with median speeds based on coordinates (X, Y)
for i, df in enumerate(dfs):
    dfs[i] = pd.merge(df[['X', 'Y', 'velocity']], median_df[['X', 'Y', 'median_velocity']], on=['X', 'Y'])

# Calculate the speed increase between consecutive videos
speed_increases = []
for i in range(1, len(dfs)):
    speed_increase = dfs[i].copy()
    speed_increase['increase'] = dfs[i]['velocity'] - dfs[i-1]['velocity']
    speed_increase['video'] = i  # Label each transition
    speed_increases.append(speed_increase)

# Concatenate all speed increases into one DataFrame
combined_increases = pd.concat(speed_increases)

# Calculate and print the average and 95th percentile speed of all the coordinates per video (first 5 videos)
avg_speeds = []
percentile_95_speeds = []
for i, df in enumerate(dfs):
    avg_speed = df['velocity'].mean()
    percentile_95_speed = np.percentile(df['velocity'], 95)
    avg_speeds.append(avg_speed)
    percentile_95_speeds.append(percentile_95_speed)
    print(f'Average speed for video {i+1}: {avg_speed:.2f} m/s, 95th percentile: {percentile_95_speed:.2f} m/s')

# Plotting
plt.figure(figsize=(9, 6))  # Set figsize to 9 by 6
colors = ['red', 'green', 'blue', 'purple']  # Define colors for each video transition

# Loop to plot 50% of data points and trendlines
for i in range(1, 5):
    video_data = combined_increases[combined_increases['video'] == i]
    
    # Sample 50% of the data points
    sampled_data = video_data.sample(frac=0.5, random_state=1)
    
    # Scatter plot for the sampled data points with smaller dot size (s=1)
    plt.scatter(sampled_data['median_velocity'], sampled_data['increase'], 
                label=f'Video {i} to {i+1}', color=colors[i-1], alpha=0.6, s=1)
    
    # Linear regression for trendline using all data points
    slope, intercept, r_value, p_value, std_err = stats.linregress(video_data['median_velocity'], video_data['increase'])
    
    # Generate trendline values
    x_vals = np.array([0, 4.5])  # Limit the x-axis range from 0 to 4.5
    y_vals = intercept + slope * x_vals

    # Plot a white line behind the dashed trendline
    plt.plot(x_vals, y_vals, color='white', linewidth=3)  # White background line
    
    # Plot the dashed trendline on top
    plt.plot(x_vals, y_vals, '--', color=colors[i-1], label=f'Trend (P-value={p_value:.2f})')

# Plot 95th percentile speed as horizontal lines for each video
#for i in range(5):
    #plt.axhline(y=percentile_95_speeds[i], color=colors[i % 4], linestyle=':', label=f'95th Percentile Video {i+1}')

# Set x-axis and y-axis limits
plt.xlim(0, 4.5)
plt.ylim(-3, 3)

plt.title('Velocity Increases vs Median Speeds')
plt.xlabel('Median Velocity (m/s)')
plt.ylabel('Velocity Increase (m/s)')

# Position the legend outside the plot on the right
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(True)

# Save the plot as a PDF
plt.savefig('speed_increase_vs_median_speed_with_trendlines_50percent_filled_background.pdf', bbox_inches='tight')

# Show the plot
#plt.show()

# Create a new figure for the boxplots
plt.figure(figsize=(9, 6))

# Define bins for median velocity (e.g., 0-0.5, 0.5-1.0, etc.)
bins = np.arange(0, 5, 0.5)  # 0 to 5 m/s in 0.5 m/s steps
labels = [f'{round(b,1)}-{round(b+0.5,1)}' for b in bins[:-1]]

# Bin the median velocities
combined_increases['velocity_bin'] = pd.cut(combined_increases['median_velocity'], bins=bins, labels=labels, include_lowest=True)

# Plot boxplots
sns.boxplot(x='velocity_bin', y='increase', data=combined_increases, palette='coolwarm')

#plt.title('Velocity Increases by Initial Velocity Bins')
plt.xlabel('Initial Median Velocity Bin (m/s)')
plt.ylabel('Velocity Increase (m/s)')
plt.ylim(-3, 3)
plt.xticks(rotation=45)
plt.yticks(np.arange(-3, 4, 1))
plt.grid(True)

# Save this new figure
plt.savefig('speed_increase_boxplots_by_velocity_bin.pdf', bbox_inches='tight')
