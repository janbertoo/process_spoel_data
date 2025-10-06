import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plotting the PDF for velocity_log with KDE
plt.figure(figsize=(8, 4))
sns.kdeplot(total_velocity_df['velocity_log'], fill=True, color='tab:blue', bw_adjust=0.5)
plt.title('PDF of Log Velocities')
plt.xlabel('Log Velocity (m/s)')
plt.ylabel('Density')
plt.xlim(0, 8)
plt.tight_layout()
plt.savefig('step9_dataframe/velocity_log_pdf_filtered.png')
plt.savefig('step9_dataframe/velocity_log_pdf_filtered.pdf')
plt.show()

