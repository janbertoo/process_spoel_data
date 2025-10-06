import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of video numbers to iterate through
vidnumbers = [1, 2, 3, 4, 5]

# Initialize an empty DataFrame to collect all velocity_log values
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

    velocity_data = []

    # Iterate over the dataframe to find pairs of rows with the same 'ID' and consecutive timestamps
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.3:
            # Calculate log velocity
            velocity_log = (np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y'])**2) ) / 0.25

            # Only append if velocity_log is below the threshold of 10 m/s
            if velocity_log <= 10:
                velocity_data.append({
                    'vid_number': vidnumber,
                    'ID': current_row['ID'],
                    'velocity_log': velocity_log
                })

    # Create a new dataframe from the collected velocity data
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

# Plotting the histogram for velocity_log with filtering applied
plt.figure(figsize=(5, 4))
plt.hist(total_velocity_df['velocity_log'], bins=30, color='tab:blue', edgecolor='black', alpha=0.7)
#plt.title('Histogram of Log Velocities')
plt.xlabel('Wood Piece Velocity (m/s)')
plt.ylabel('Frequency')
plt.xlim(0,8)

# Add mean and median lines
mean_velocity = total_velocity_df['velocity_log'].mean()
median_velocity = total_velocity_df['velocity_log'].median()
plt.axvline(mean_velocity, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_velocity:.2f} m/s')
plt.axvline(median_velocity, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_velocity:.2f} m/s')
plt.legend()

plt.tight_layout()
plt.savefig('step9_dataframe/velocity_log_histogram_filtered_smaller.png')
plt.savefig('step9_dataframe/velocity_log_histogram_filtered_smaller.pdf')
#plt.show()

# Calculate the percentage of velocities between 3 and 4 m/s
between_3_and_4 = total_velocity_df[(total_velocity_df['velocity_log'] >= 2.5) & (total_velocity_df['velocity_log'] <= 4.5)]
percentage = (len(between_3_and_4) / len(total_velocity_df)) * 100

print(f"Percentage of velocities between 3 and 4 m/s: {percentage:.2f}%")
