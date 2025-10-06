import os

old_extension = 'labels'
new_extension = 'labels_no_confidence'

folders = [
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid1/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid2/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid3/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid4/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone1/drone1vid5/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone6/drone6vid1/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone6/drone6vid2/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone6/drone6vid3/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone6/drone6vid4/labels',
	'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone6/drone6vid5/labels',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone16/drone16vid1/labels',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone16/drone16vid2/labels',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone16/drone16vid3/labels',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone16/drone16vid4/labels',
	#'/home/jean-pierre/ownCloud/phd/code_code_code_code_code/process_spoel_data/cut_data/DCIM-drone16/drone16vid5/labels'
]

for old_folder in folders:
	new_folder = old_folder.replace(old_extension,new_extension)

	if os.path.exists(new_folder) == False:
		os.mkdir(new_folder)
		print('create folder')

	for file in os.listdir(old_folder):
		old_file = os.path.join(old_folder,file)
		new_file = os.path.join(new_folder,file)

		#read txt file
		with open(old_file) as f:
			lines = f.readlines()

		with open(new_file, "w") as new_f:

			for line in lines:
				line_split = line.split(' ')
				new_f.write(line_split[0]+' '+line_split[1]+' '+line_split[2]+' '+line_split[3]+' '+line_split[4]+'\n')
