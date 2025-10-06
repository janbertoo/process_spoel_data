import geopandas as gpd
from shapely.geometry import Point

shapefile_path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/create_geotiff/vid2.shp'

def is_coordinate_within_shapefile(longitude, latitude, shapefile_path):
    # Create a Point object for the given coordinate
    point = Point(longitude, latitude)
    
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    
    # Check if the point is within any of the geometries in the shapefile
    is_within = gdf.geometry.contains(point).any()
    
    return is_within

# Example usage
X = 180
Y = 80
#shapefile_path = "path/to/your/shapefile.shp"

print(X,Y)
result = is_coordinate_within_shapefile(Y, X, shapefile_path)
print(result)