import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

vidnumbers = [1, 2, 3, 4, 5]

# Prepare an empty DataFrame to store all combined data
total_velocity_df = pd.DataFrame()

for vidnumber in vidnumbers:
    # Load the data for each video
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
    
    df = pd.DataFrame(data)

    # Ensure columns 'center_geo_x', 'center_geo_y', and 'timestamp' are numeric
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
            velocity_log = ( np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y'])**2) ) / 0.25

            # Calculate midpoint
            mid_x = (current_row['center_geo_x'] + next_row['center_geo_x']) / 2
            mid_y = (current_row['center_geo_y'] + next_row['center_geo_y']) / 2

            # Find the closest flow velocity using KDTree
            _, idx = kdtree.query([mid_x, mid_y])
            closest_flow_velocity = float(flow_velocity_df.iloc[idx]['velocity'])  # Ensure it's a float

            # Calculate normalised velocity
            normalised_velocity = velocity_log - closest_flow_velocity

            # Append the result to the list
            velocity_data.append({
                'vid_number': vidnumber,
                'ID': current_row['ID'],
                'length_median': current_row['length_median'],
                'diameter_median': current_row['diameter_median'],
                'volume_median': current_row['volume_median'],
                'velocity_log': velocity_log,
                'closest_flow_velocity': closest_flow_velocity,
                'normalised_velocity': normalised_velocity
            })

    # Create a new dataframe from the collected velocity data
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

# Drop any rows with missing values
total_velocity_df = total_velocity_df.dropna(subset=['normalised_velocity', 'length_median', 'diameter_median', 'volume_median'])

# Build the multiple linear regression model
X = total_velocity_df[['length_median', 'diameter_median', 'volume_median']]  # Independent variables
y = total_velocity_df['normalised_velocity']  # Dependent variable

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

# If needed, plot the residuals
plt.figure(figsize=(6, 4))
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.tight_layout()
plt.show()
