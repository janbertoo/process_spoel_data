import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import ast  # To safely evaluate string as list

# Base coordinates
Xbase = 2803987
Ybase = 1175193

# Load the shapefile
shapefile_path = 'diverge_shapefiles/diverge.shp'
polygons = gpd.read_file(shapefile_path)

# Helper function to parse coordinates
def parse_coordinates(coord_str):
    try:
        return np.array(ast.literal_eval(coord_str))
    except (ValueError, SyntaxError):
        return None  # Handle invalid or missing data

# Helper function to calculate angle
def calculate_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(min(angle, np.pi - angle))

vidnumbers = [1, 2, 3, 4, 5]
total_velocity_df = pd.DataFrame()

for vidnumber in vidnumbers:
    # Load the data for each video
    with open(f'step9_dataframe/dataframe_vid{vidnumber}_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data)

    # Adjust coordinates to the same system as the shapefile
    df['center_geo_x'] += Xbase
    df['center_geo_y'] += Ybase

    # Add geometry to classify data points
    geometry = [Point(xy) for xy in zip(df['center_geo_x'], df['center_geo_y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Classify points as "diverging" or "converging"
    gdf['class'] = gdf['geometry'].apply(
        lambda point: 'diverging' if any(polygon.contains(point) for polygon in polygons.geometry) else 'converging'
    )

    velocity_data = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.3:
            # Calculate log velocity
            velocity_log = (
                np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 +
                        (next_row['center_geo_y'] - current_row['center_geo_y'])**2)
            ) / 0.25

            # Parse corner coordinates
            if current_row['orientation'] == 'tlbr':
                top_left = parse_coordinates(current_row['top_left_geo'])
                bottom_right = parse_coordinates(current_row['bottom_right_geo'])
                log_vector = bottom_right - top_left if top_left is not None and bottom_right is not None else None
            else:
                top_right = parse_coordinates(current_row['top_right_geo'])
                bottom_left = parse_coordinates(current_row['bottom_left_geo'])
                log_vector = bottom_left - top_right if top_right is not None and bottom_left is not None else None

            if log_vector is None:
                continue  # Skip rows with invalid coordinates

            # Calculate log angle
            flow_vector = np.array([1, 0])  # Example flow vector
            log_angle = calculate_angle(log_vector, flow_vector)

            velocity_data.append({
                'log_angle': log_angle,
                'class': current_row['class'],
            })

    # Create a new dataframe from the velocity data
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

print(total_velocity_df)
# Create histograms for diverging and converging classes
for classification in ['diverging', 'converging']:
    subset = total_velocity_df[total_velocity_df['class'] == classification]
    plt.hist(subset['log_angle'], bins=np.arange(0, 100, 10), alpha=0.6, label=classification)

plt.title('Log Angle Distribution by Class')
plt.xlabel('Log Angle (degrees)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()

# Save the histograms
plt.savefig('log_angle_distribution_by_class.png')
plt.savefig('log_angle_distribution_by_class.pdf')

plt.show()

