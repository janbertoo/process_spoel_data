import os

curdir = os.getcwd()
print(curdir)
#print(os.path.join(curdir+'/cut_data/DCIM-drone1/drone1vid1/labels_with_gcps'))
X = 3840
Y = 2160

percentage = 2

paths = [
	curdir+'/cut_data/DCIM-drone1/drone1vid1/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone1/drone1vid2/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone1/drone1vid3/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone1/drone1vid4/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone1/drone1vid5/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone6/drone6vid1/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone6/drone6vid2/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone6/drone6vid3/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone6/drone6vid4/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone6/drone6vid5/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone16/drone16vid1/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone16/drone16vid2/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone16/drone16vid3/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone16/drone16vid4/labels_with_gcps',
	curdir+'/cut_data/DCIM-drone16/drone16vid5/labels_with_gcps'
]

newlabelsfolder = 'labels_with_gcps_no2percent'

def checkTxtsFor2percentAndDeleteAndCopy(path):
	newlabelspath = path.replace('labels_with_gcps',newlabelsfolder)

	if os.path.exists(newlabelspath) == False:
		os.mkdir(newlabelspath)

	for file in os.listdir(path):
		if file[-4:] == '.txt':
			
			fullfilepath_txt = os.path.join(path,file)
			new_fullfilepath_txt = fullfilepath_txt.replace('labels_with_gcps',newlabelsfolder)
			
			with open(fullfilepath_txt) as f:
				lines = f.readlines()
				lines_to_store = []
			
			for line in lines:
				line_split = (line.split(',')[0]).split(' ')

				if ( ( percentage / 100 ) * X ) < float(line_split[0]) < ( ( ( 100 - percentage ) / 100 ) * X ) and ( ( percentage / 100 ) * Y ) < float(line_split[1]) < ( ( ( 100 - percentage ) / 100 ) * Y ):
					lines_to_store.append(line)

			with open(new_fullfilepath_txt, "w") as new_f:
				for line in lines_to_store:
					new_f.write(line)

for path in paths:
	print(path)
	checkTxtsFor2percentAndDeleteAndCopy(path)