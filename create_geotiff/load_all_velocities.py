import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

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

df_drone1 = pd.read_excel(velocities_path_drone1)
df_drone6 = pd.read_excel(velocities_path_drone6)
df_drone16 = pd.read_excel(velocities_path_drone16)

def load_xlsx_files(file_pattern):
    # Use glob to find all files matching the pattern
    files = glob.glob(file_pattern)
    
    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Loop through each file and read it into a DataFrame
    for file in files:
        df = pd.read_excel(file)
        dfs.append(df)

    # Concatenate all DataFrames into one big DataFrame
    result_df = pd.concat(dfs, ignore_index=True)
    
    return result_df

# Specify the file pattern for your Excel files
file_pattern = os.path.join(curdir,'velocities')+"/*.xlsx"

# Call the function to load the Excel files into one DataFrame
big_dataframe = load_xlsx_files(file_pattern)

# Display the resulting DataFrame
print(big_dataframe)


big_dataframe["Norme"].plot(kind = 'kde')
#big_dataframe.plot('Norme',kind = 'bar')
ax = plt.gca()
ax.set_xlim([0, 7])
plt.savefig('norme.jpg')



# Assuming 'your_dataframe' is the name of your DataFrame and 'column_name' is the name of the column you want to filter
threshold_value = 7  # Set your threshold value

# Filtering rows based on the condition
filtered_dataframe = big_dataframe[big_dataframe['Norme'] <= threshold_value]

print(filtered_dataframe)

filtered_dataframe["Norme"].plot(kind = 'kde')
#big_dataframe.plot('Norme',kind = 'bar')
#ax = plt.gca()
#ax.set_xlim([0, 7])
plt.savefig('norme_filtered.jpg')


# Creating a DataFrame with the removed rows
removed_rows_dataframe = big_dataframe[big_dataframe['Norme'] > threshold_value]

# Now, 'removed_rows_dataframe' contains the rows that were removed
# Print the removed rows
print("Removed Rows:")
print(removed_rows_dataframe)



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

# Print the generated coordinates
for coord in coordinates:
	result = is_coordinate_within_shapefile(coord[0], coord[1], shapefile_path)
	if result == True:
		print(coord)
		print(result)

