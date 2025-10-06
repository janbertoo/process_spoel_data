import os
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry import LineString
import matplotlib.pyplot as plt

curdir = os.getcwd()

gps_dir = 'GPS_Data-20230619T123957Z-001/GPS_Data'

csv_waterlines_pre = os.path.join(curdir,gps_dir,'presections_waterlines.csv')
csv_waterlines_post = os.path.join(curdir,gps_dir,'post_waterlines.csv')

lines_along_river = [(2804018,1175234),(2804173,1175273)]
lines_along_river = [(2804090.912822751,1175252.3458070147),(2804173,1175273)]

line = LineString([lines_along_river[0], lines_along_river[1]])

pre = pd.read_csv(csv_waterlines_pre)
post = pd.read_csv(csv_waterlines_post)

pre['coordinate_2d'] = pre.apply(lambda row: (row.xcoor,row.ycoor) , axis = 1)
post['coordinate_2d'] = post.apply(lambda row: (row.xcoor,row.ycoor) , axis = 1)


def getProjection(line,point):
	point = Point(point)
	x = np.array(point.coords[0])

	u = np.array(line.coords[0])
	v = np.array(line.coords[len(line.coords)-1])

	n = v - u
	n /= np.linalg.norm(n, 2)

	P = u + n*np.dot(x - u, n)
	#print(P) #0.2 1.
	return(P)


pre['projected_point'] = pre.apply(lambda row: getProjection(line,row.coordinate_2d) , axis = 1)
post['projected_point'] = post.apply(lambda row: getProjection(line,row.coordinate_2d) , axis = 1)


def calc_distance(lines_along_river,point_to_project):
	#print(lines_along_river)
	#print(point_to_project)
	if point_to_project[0] < lines_along_river[0][0]:
		distance = - np.sqrt( ( point_to_project[0] - lines_along_river[0][0] ) ** 2 + ( point_to_project[1] - lines_along_river[0][1] ) ** 2 )
	else:
		distance = np.sqrt( ( point_to_project[0] - lines_along_river[0][0] ) ** 2 + ( point_to_project[1] - lines_along_river[0][1] ) ** 2 )

	return(distance)



#pre['projected_point_distance'] = pre.apply(lambda row: np.sqrt( ( row.projected_point[0] - lines_along_river[0][0] ) ** 2 + ( row.projected_point[1] - lines_along_river[0][1] ) ** 2 ) , axis = 1)
#post['projected_point_distance'] = post.apply(lambda row: np.sqrt( ( row.projected_point[0] - lines_along_river[0][0] ) ** 2 + ( row.projected_point[1] - lines_along_river[0][1] ) ** 2 ) , axis = 1)

pre['projected_point_distance'] = pre.apply(lambda row: calc_distance(lines_along_river,row.projected_point) , axis = 1)
post['projected_point_distance'] = post.apply(lambda row: calc_distance(lines_along_river,row.projected_point) , axis = 1)

pre = pre.sort_values(by='projected_point_distance')
post = post.sort_values(by='projected_point_distance')

print(pre)
print(post)

#fit line though points
x = post['projected_point_distance']
y = post['zcoor']
a, b = np.polyfit(x, y, 1)
print(a,b)

df_all = pd.concat([pre, post], axis=0)

plot = df_all.plot.line(x='projected_point_distance',y='zcoor', figsize=(10,6))

fig = plot.get_figure()
fig.savefig("output_pre_post.png")









def get_corrected_zcoor(xcoor,ycoor,zcoor):

	lines_along_river = [(2804090.912822751,1175252.3458070147),(2804173,1175273)]
	waterlevel_slope = 0.01437202672278633

	point = Point(xcoor,ycoor)
	x = np.array(point.coords[0])

	u = np.array(line.coords[0])
	v = np.array(line.coords[len(line.coords)-1])

	n = v - u
	n /= np.linalg.norm(n, 2)

	point_to_project = u + n*np.dot(x - u, n)
	
	if point_to_project[0] < lines_along_river[0][0]:
		distance = - np.sqrt( ( point_to_project[0] - lines_along_river[0][0] ) ** 2 + ( point_to_project[1] - lines_along_river[0][1] ) ** 2 )
	else:
		distance = np.sqrt( ( point_to_project[0] - lines_along_river[0][0] ) ** 2 + ( point_to_project[1] - lines_along_river[0][1] ) ** 2 )

	correction = -distance * waterlevel_slope

	return(correction)

