import os

X = 3840
Y = 2160

paths = [
	'/home/jean-pierre/Desktop/18dec/DCIM-drone6/drone6vid1',
	'/home/jean-pierre/Desktop/18dec/DCIM-drone6/drone6vid2',
	'/home/jean-pierre/Desktop/18dec/DCIM-drone6/drone6vid3',
	'/home/jean-pierre/Desktop/18dec/DCIM-drone6/drone6vid4',
	'/home/jean-pierre/Desktop/18dec/DCIM-drone6/drone6vid5',
]

def mergeGCPS(path):
	GCPloc = '2804110.298 1175277.929 1490'

	labelspath = os.path.join(path,'labels')
	VIRTUALlabelspath = os.path.join(path,'labels_bloem2')
	GCPSlabelspath = os.path.join(path,'labels_with_gcps')

	txtlist = []
	
	for file in os.listdir(VIRTUALlabelspath):
		if file[-4:] == '.txt':
			txtlist.append(file)
	
	txtlist.sort()

	for txt in txtlist:
		labels_txt = os.path.join(labelspath,txt)
		VIRTUAL_txt = os.path.join(VIRTUALlabelspath,txt)
		GCPSlabels_txt = os.path.join(GCPSlabelspath,txt)
		#print(VIRTUAL_txt)
		#read virtual GCP txt file
		with open(VIRTUAL_txt) as f:
			lines = f.readlines()

			line_to_add_to_labels = lines[0]
			
			line_split = lines[0].split(' ')
			#print(line)

			line_to_add_to_GCPS_label = str( float(line_split[1]) * X ) + ' ' + str( float(line_split[2]) * Y ) + ', ' + GCPloc

			print(line_to_add_to_GCPS_label)

			#write to labels
			with open(labels_txt, "a") as text_file_labels:
				text_file_labels.write(line_to_add_to_labels)

			#write to GCPS labels
			with open(GCPSlabels_txt, "a") as text_file_GCPS_labels:
				text_file_GCPS_labels.write(line_to_add_to_GCPS_label)





#mergeGCPS(paths[0])

for path in paths:
	mergeGCPS(path)