import os

x = 3840
y = 2160

folders = [
	'/home/jean-pierre/Desktop/5jan/move_labels/vidzoveel'
]

#for rock23
folders = [
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid1',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid2',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid3',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid4',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid5',
]



newfolder = 'labels_rock23'
#newfolder = 'labels_tree12'

location_rock2 = ['2804122.82171403','1175243.20813898','1490'] 	#8
location_rock3 = ['2804123.92670381','1175286.55109553','1490']		#9
location_tree1 = ['2804083.47260533','1175251.52298227','1490']		#10
location_tree2 = ['2804076.03330728','1175274.29283175','1490']		#11

for folder in folders:
	labels_only_folder =		os.path.join(folder,'labels')
	labels_with_gcps_folder =	os.path.join(folder,'labels_with_gcps')
	labels_new_folder =			os.path.join(folder,newfolder)

	for file in os.listdir(labels_new_folder):
		if file[-4:] == '.txt':
			labels_only_path = os.path.join(labels_only_folder,file)
			labels_with_gcps_path = os.path.join(labels_with_gcps_folder,file)
			labels_new_path = os.path.join(labels_new_folder,file)

			with open(labels_new_path) as f_new:
				lines = f_new.readlines()

			rw_location = None

			for i in range(2):
				split_line = lines[i].split(' ')
				
				if split_line[0] == '8':
					rw_location = location_rock2
				if split_line[0] == '9':
					rw_location = location_rock3
				if split_line[0] == '10':
					rw_location = location_tree1
				if split_line[0] == '11':
					rw_location = location_tree2

				if rw_location != None:
					f_labels_only = open(labels_only_path, 'a')
					f_labels_only.write(split_line[0]+' '+split_line[1]+' '+split_line[2]+' '+split_line[3]+' '+split_line[4]+' '+split_line[5])

					f_labels_with_gcps = open(labels_with_gcps_path, 'a')
					xcoor = str( float(split_line[1]) * x )
					ycoor = str( float(split_line[2]) * y )
					f_labels_with_gcps.write(xcoor+' '+ycoor+', '+rw_location[0]+' '+rw_location[1]+' '+rw_location[2]+'\n')