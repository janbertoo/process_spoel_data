import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from tabulate import tabulate
import pickle
from itertools import combinations
from geopy.distance import geodesic

savefigfolder = 'step9_dataframe/'
file_path = os.path.join(savefigfolder,'dataframe_vidALL.p')
df = pd.read_pickle(file_path)
print(df)



# Function to convert center_geo values to tuples of floats
def parse_geo(geo_str):
    try:
        return tuple(map(float, geo_str.strip('[]').split(',')))
    except Exception as e:
        print(f"Error parsing geo coordinates {geo_str}: {e}")
        return None

# Convert center_geo values
df['center_geo'] = df['center_geo'].apply(parse_geo)

# Remove rows with invalid center_geo values
df = df.dropna(subset=['center_geo'])

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    try:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except Exception as e:
        print(f"Error calculating distance between {point1} and {point2}: {e}")
        return 0

# Group by 'vidnumber' and 'ID' and calculate the required metrics
grouped = df.groupby(['vidnumber', 'ID']).apply(lambda group: pd.Series({
    'count': len(group),
    'volume_median': group['volume_median'].iloc[0],
    'max_distance': max(calculate_distance(p1, p2) for p1, p2 in combinations(group['center_geo'], 2)) if len(group) > 1 else 0
})).reset_index()

# Calculate the ratio of count to max_distance, handling division by zero
grouped['count_distance_ratio'] = grouped.apply(lambda row: row['count'] / row['max_distance'] if row['max_distance'] != 0 else float('inf'), axis=1)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(grouped['count_distance_ratio'],grouped['volume_median'], marker='o')
plt.xlabel('Volume Median')
plt.ylabel('Count/Max Distance')
plt.title('Volume Median vs. Count/Max Distance for each Unique vidnumber-ID Combination')
plt.grid(True)
plt.show()