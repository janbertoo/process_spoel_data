from scipy.interpolate import griddata
import numpy as np

closest_velocities = np.array([[-1.52919707,-1.06369582],[-1.63347168,-1.12211713],[0.03223598,-0.01630312],[-0.03467784,-0.0563487]])
closest_coordinates = np.array([[51.3998,33.4437],[52.0948,33.8746],[51.9975,32.7348],[50.6909,33.0128]])
print(closest_coordinates)
print(closest_velocities)
print(closest_velocities[:, 1])

closest_velocities = np.array([[1,1],[0,0],[0,0],[0,0]])
closest_coordinates = np.array([[1,1],[100,100],[3,4],[200,33.0128]])


X = 51.5
Y = 33.5

X = 2
Y = 2
new_coordinate = (X,Y)
print(new_coordinate)

interpolated_vx = griddata(closest_coordinates, closest_velocities[:, 0], new_coordinate, method='linear')
interpolated_vy = griddata(closest_coordinates, closest_velocities[:, 1], new_coordinate, method='linear')

print(interpolated_vx)
print(interpolated_vy)