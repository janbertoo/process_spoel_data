import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import pickle

curdir = os.getcwd()

#vidnumber = 1
threshold = 0.3 #30 percent confidence

meters_of_width_detection_zone = 1.5#0.5 #m total width
framewidth = 3

#define base coordinates
Xbase = 2803987
Ybase = 1175193

x_width = 3840
y_height = 2160

#inferred_txt_files_path = os.path.join(curdir,'vid'+str(vidnumber)+'/inferred_rw_coordinates')

section_1_line = [ [ 2804180.788 , 1175253.469 ] , [ 2804162.401 , 1175290.274 ] ] #from drone 1
section_2_line = [ [ 2804141.918 , 1175249.764 ] , [ 2804132.931 , 1175280.158 ] ] #from drone 1	#section AA
section_3_line = [ [ 2804108.578 , 1175234.677 ] , [ 2804104.240 , 1175273.729 ] ] #from drone 6	#section BB
section_4_line = [ [ 2804072.864 , 1175227.535 ] , [ 2804056.896 , 1175273.277 ] ] #from drone 6	#section CC
section_5_line = [ [ 2804059.828 , 1175226.782 ] , [ 2804030.929 , 1175268.910 ] ] #from drone 16
section_6_line = [ [ 2804041.128 , 1175221.571 ] , [ 2804001.912 , 1175252.464 ] ] #from drone 16




def find_wood_in_section_rectangles(section_line,coordinate_wood):
	'''
	line_corners = [ 
		[ section_line[0][0] - meters_of_width_detection_zone , section_line[0][1] ] ,
		[ section_line[0][0] + meters_of_width_detection_zone , section_line[0][1] ] ,
		[ section_line[1][0] - meters_of_width_detection_zone , section_line[1][1] ] ,
		[ section_line[1][0] + meters_of_width_detection_zone , section_line[1][1] ]
	]
	'''
	
	x, y = coordinate_wood[0],coordinate_wood[1]

	lons_vect = [ 
		section_line[0][0] - meters_of_width_detection_zone ,
		section_line[0][0] + meters_of_width_detection_zone ,
		section_line[1][0] - meters_of_width_detection_zone ,
		section_line[1][0] + meters_of_width_detection_zone
	]
	
	lats_vect = [ 
		section_line[0][1] ,
		section_line[0][1] ,
		section_line[1][1] ,
		section_line[1][1] 
	]

	#np.array([[Lon_A, Lat_A], [Lon_B, Lat_B], [Lon_C, Lat_C], [Lon_D, Lat_D]])

	lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
	polygon = Polygon(lons_lats_vect) # create polygon
	point = Point(x,y) # create point
	#print(polygon.contains(point)) # check if polygon contains point
	#print(point.within(polygon)) # check if a point is in the polygon 

	return(point.within(polygon))

#print(find_wood_in_section_rectangles(section_1_line,(2804180.788 , 1175253.569)))
#print(find_wood_in_section_rectangles(section_1_line,(2804180.788 , 1175253.369)))

#output_f.write('0 '+str(X[i])+' '+str(Y[i])+' '+str(TL_X[i])+' '+str(TL_Y[i])+' '+str(BR_X[i])+' '+str(BR_Y[i])+' '+conf_list[i])

woods_in_section_1 = []
woods_in_section_2 = []
woods_in_section_3 = []
woods_in_section_4 = []
woods_in_section_5 = []
woods_in_section_6 = []

wood_lengths = []


for i in range(5):
	inferred_txt_files_path = os.path.join(curdir,'vid'+str(i+1)+'/inferred_rw_coordinates')

	txtfiles = []
	for file in os.listdir(inferred_txt_files_path):
		if file[-4:] == '.txt':
			txtfiles.append(file)

	txtfiles.sort()
	#print(txtfiles)
	for file in txtfiles:
		#print(file)
		drone_number = int((file.split('DCIM-drone')[-1]).split('_drone')[0])
		with open(os.path.join(inferred_txt_files_path,file)) as txt_file:
			lines = txt_file.readlines()

		for line in lines:
			line_split = line.split(' ')

			length_bbox = np.sqrt( (float(line_split[3]) - float(line_split[5])) ** 2 + (float(line_split[4]) - float(line_split[6])) ** 2 ) #in meters

			remembered_frame_number_sec1 = -10
			remembered_frame_number_sec2 = -10
			remembered_frame_number_sec3 = -10
			remembered_frame_number_sec4 = -10
			remembered_frame_number_sec5 = -10
			remembered_frame_number_sec6 = -10

			#only use the detection above a certain threshold and above 1 meter in length (Large Wood)
			if float(line_split[-1]) > threshold and length_bbox > 1:
				wood_lengths.append(length_bbox)
				if drone_number == 1:
					if find_wood_in_section_rectangles( section_1_line , ( float(line_split[1])+Xbase , float(line_split[2])+Ybase ) ) == True:
					 	woods_in_section_1.append([((file.split('vid')[-1]).split('_')[0])+'_'+(file.split('frame')[1]).split('_DCIM')[0],file,line])
					if find_wood_in_section_rectangles( section_2_line , ( float(line_split[1])+Xbase , float(line_split[2])+Ybase ) ) == True:
						woods_in_section_2.append([((file.split('vid')[-1]).split('_')[0])+'_'+(file.split('frame')[1]).split('_DCIM')[0],file,line])
				if drone_number == 6:
					if find_wood_in_section_rectangles( section_3_line , ( float(line_split[1])+Xbase , float(line_split[2])+Ybase ) ) == True:
						woods_in_section_3.append([((file.split('vid')[-1]).split('_')[0])+'_'+(file.split('frame')[1]).split('_DCIM')[0],file,line])
					if find_wood_in_section_rectangles( section_4_line , ( float(line_split[1])+Xbase , float(line_split[2])+Ybase ) ) == True:
						woods_in_section_4.append([((file.split('vid')[-1]).split('_')[0])+'_'+(file.split('frame')[1]).split('_DCIM')[0],file,line])
				if drone_number == 16:
					if find_wood_in_section_rectangles( section_5_line , ( float(line_split[1])+Xbase , float(line_split[2])+Ybase ) ) == True:
						woods_in_section_5.append([((file.split('vid')[-1]).split('_')[0])+'_'+(file.split('frame')[1]).split('_DCIM')[0],file,line])
					if find_wood_in_section_rectangles( section_6_line , ( float(line_split[1])+Xbase , float(line_split[2])+Ybase ) ) == True:
						woods_in_section_6.append([((file.split('vid')[-1]).split('_')[0])+'_'+(file.split('frame')[1]).split('_DCIM')[0],file,line])


file = open(os.path.join(curdir,'woods_in_section_1.p'), 'wb')
pickle.dump(woods_in_section_1, file)
file = open(os.path.join(curdir,'woods_in_section_2.p'), 'wb')
pickle.dump(woods_in_section_2, file)
file = open(os.path.join(curdir,'woods_in_section_3.p'), 'wb')
pickle.dump(woods_in_section_3, file)
file = open(os.path.join(curdir,'woods_in_section_4.p'), 'wb')
pickle.dump(woods_in_section_4, file)
file = open(os.path.join(curdir,'woods_in_section_5.p'), 'wb')
pickle.dump(woods_in_section_5, file)
file = open(os.path.join(curdir,'woods_in_section_6.p'), 'wb')
pickle.dump(woods_in_section_6, file)


tot = 0
for wood_length in wood_lengths:
	tot = tot + wood_length
average = tot / len(wood_lengths)

print(average)




print(' ')
print(' ')
woods_in_section_1.sort()
print(woods_in_section_1)
print(' ')
print(len(woods_in_section_1))
print(' ')
print(' ')
woods_in_section_2.sort()
print(woods_in_section_2)
print(' ')
print(len(woods_in_section_2))
print(' ')
print(' ')
woods_in_section_3.sort()
print(woods_in_section_3)
print(' ')
print(len(woods_in_section_3))
print(' ')
print(' ')
woods_in_section_4.sort()
print(woods_in_section_4)
print(' ')
print(len(woods_in_section_4))
print(' ')
print(' ')
woods_in_section_5.sort()
print(woods_in_section_5)
print(' ')
print(len(woods_in_section_5))
print(' ')
print(' ')
woods_in_section_6.sort()
print(woods_in_section_6)
print(' ')
print(len(woods_in_section_6))
