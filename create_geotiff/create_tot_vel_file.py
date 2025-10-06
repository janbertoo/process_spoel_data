import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import csv

vidnumber = 1

shapefile_path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/vid2.shp'
curdir = os.getcwd()

velocities_path_drone1 = os.path.join(curdir,'velocities_vid'+str(vidnumber),'drone1vid'+str(vidnumber)+'.xlsx')
velocities_path_drone6 = os.path.join(curdir,'velocities_vid'+str(vidnumber),'drone6vid'+str(vidnumber)+'.xlsx')
velocities_path_drone16 = os.path.join(curdir,'velocities_vid'+str(vidnumber),'drone16vid'+str(vidnumber)+'.xlsx')

def is_coordinate_within_shapefile(longitude, latitude, shapefile_path):
    # Create a Point object for the given coordinate
    point = Point(longitude, latitude)
    
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    
    # Check if the point is within any of the geometries in the shapefile
    is_within = gdf.geometry.contains(point).any()

    return is_within

# Function to find the closest 4 coordinates and interpolate velocities
def interpolate_velocities(df, new_coordinate):
    # Extract coordinates and velocities from DataFrame
    coordinates = df[['X', 'Y']].values
    velocities = df[['Vx', 'Vy']].values

    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(coordinates)

    # Find the indices of the 4 closest coordinates to the new coordinate
    _, indices = tree.query(new_coordinate, k=4)

    # Extract the coordinates and velocities of the 4 closest points
    closest_coordinates = coordinates[indices]
    closest_velocities = velocities[indices]

    # Interpolate velocities for the new coordinate using griddata
    interpolated_vx = griddata(closest_coordinates, closest_velocities[:, 0], new_coordinate, method='linear')
    interpolated_vy = griddata(closest_coordinates, closest_velocities[:, 1], new_coordinate, method='linear')

    return interpolated_vx, interpolated_vy


#define base coordinates
Xbase = 2803987
Ybase = 1175193

#gcp 6 and 9
points_line1 = ((2804101.6-Xbase,1175278.832-Ybase),(2804107.518-Xbase,1175241.001-Ybase))

x1_line1 = points_line1[0][0]
y1_line1 = points_line1[0][1]
x2_line1 = points_line1[1][0]
y2_line1 = points_line1[1][1]

m_line1 = (y1_line1-y2_line1)/(x1_line1-x2_line1)                           #slope
b_line1 = (x1_line1*y2_line1 - x2_line1*y1_line1)/(x1_line1-x2_line1)       #y-intercept

#gcp 4 and 10
points_line2 = ((2804036.78-Xbase,1175271.824-Ybase),(2804069.799-Xbase,1175236.847-Ybase))

x1_line2 = points_line2[0][0]
y1_line2 = points_line2[0][1]
x2_line2 = points_line2[1][0]
y2_line2 = points_line2[1][1]

m_line2 = (y1_line2-y2_line2)/(x1_line2-x2_line2)
b_line2 = (x1_line2*y2_line2 - x2_line2*y1_line2)/(x1_line2-x2_line2)

df_drone1 = pd.read_excel(velocities_path_drone1)
df_drone6 = pd.read_excel(velocities_path_drone6)
df_drone16 = pd.read_excel(velocities_path_drone16)

#big_dataframe["Norme"].plot(kind = 'kde')
#big_dataframe.plot('Norme',kind = 'bar')
#ax = plt.gca()
#ax.set_xlim([0, 7])
#plt.savefig('norme.jpg')

# Assuming 'your_dataframe' is the name of your DataFrame and 'column_name' is the name of the column you want to filter
threshold_value = 7  # Set your threshold value

# Filtering rows based on the condition
filtered_df_drone1 = df_drone1[df_drone1['Norme'] <= threshold_value]
filtered_df_drone6 = df_drone6[df_drone6['Norme'] <= threshold_value]
filtered_df_drone16 = df_drone16[df_drone16['Norme'] <= threshold_value]

step_size = 0.5

# Define the ranges and step sizes
x_range = range(int(0/step_size), int(220/step_size))
y_range = range(int(25/step_size), int(100/step_size))

# Create a list to store coordinates
coordinates = []

# Generate coordinates using nested loops
for y in y_range:
    for x in x_range:
        coordinates.append([x*step_size, y*step_size])


new_coors_and_velocities = []

# Print the generated coordinates
for coord in coordinates:
	result = is_coordinate_within_shapefile(coord[0], coord[1], shapefile_path)
	if result == True:
		if coord[0] * m_line1 + b_line1 < coord[1]:
			newvx,newvy = interpolate_velocities(df_drone1,(coord[0],coord[1]))
			new_coors_and_velocities.append([coord[0],coord[1],newvx,newvy])
		if coord[0] * m_line1 + b_line1 > coord[1] and coord[0] * m_line2 + b_line2 < coord[1]:
			newvx,newvy = interpolate_velocities(df_drone6,(coord[0],coord[1]))
			new_coors_and_velocities.append([coord[0],coord[1],newvx,newvy])
		if coord[0] * m_line2 + b_line2 > coord[1]:
			newvx,newvy = interpolate_velocities(df_drone16,(coord[0],coord[1]))
			new_coors_and_velocities.append([coord[0],coord[1],newvx,newvy])

		#print(coord[0])
		#print(coord[1])
		#print(newvx)
		#print(newvy)
		#print('')


# Specify the file path
csv_file_path = os.path.join(curdir,'velocities_vid'+str(vidnumber),'all_vid'+str(vidnumber)+'.csv')

# Write the list to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write each sublist as a row in the CSV file
    for row in new_coors_and_velocities:
        csv_writer.writerow(row)