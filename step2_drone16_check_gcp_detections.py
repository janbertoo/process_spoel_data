import os
import shutil
import random

curdir = os.getcwd()

#what do we expect to see for drone 1? [wood, GCP, GCPv2, GCPv3, GCPv4]
#expect = [0, 4, 0, 0, 0] #drone1
#expect = [0, 5, 0, 0, 0] #drone6
expect = [0, 4, 1, 1, 0] #drone16

#path = '/home/jean-pierre/Desktop/destiny_folder_test/spoel_2023_cut_data'
path = os.path.join(curdir,'cut_data')

drone = 'DCIM-drone16'

dronepath = os.path.join(path,drone)
#labelpath = '/home/jean-pierre/Desktop/destiny_folder_test/spoel_2023_cut_data/DCIM-drone1/DJI_0300/labels'

#analyzelabelpath = '/home/jean-pierre/Desktop/analyse27okt4'

analyzelabelpath = os.path.join(curdir,'analysedetections')

if os.path.isdir(analyzelabelpath) != True:
	os.mkdir(analyzelabelpath)

#define paths to store data to analyse in
analysepathimages = os.path.join(analyzelabelpath,'images')
analysepathlabels = os.path.join(analyzelabelpath,'labels')

#create the directories if they d not exist
if os.path.isdir(analysepathimages) != True:
	os.mkdir(analysepathimages)
if os.path.isdir(analysepathlabels) != True:
	print('MAKE DIR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	os.mkdir(analysepathlabels)

#create list of folder that were inferred
folderlist = []

for folder in os.listdir(dronepath):
	labelfolder = os.path.join(dronepath,folder,'labels')
	folderlist.append(labelfolder)

folderlist.sort()

print('Folders to analyse: ',folderlist)





def create_txt_list_from_folder(folder):
	txtlist = []

	for file in os.listdir(folder):
		if file[-4:] == '.txt':
			filepath = os.path.join(folder,file)
			txtlist.append(filepath)

	txtlist.sort()

	return txtlist




def analyzelabelpath(labelpath,txtlist):

	#wood_count =  expect[0]
	#GCP_count = expect[1]
	#GCPv2_count = expect[2]
	#GCPv3_count = expect[3]
	#GCPv4_count = expect[4]

	nongood_txts_list = []
	nongood_jpgs_list = []
	count = 0
	#sessioncount = 0
	#mem = False

	for txt in txtlist:
		wood_count =  0
		GCP_count = 0
		GCPv2_count = 0
		GCPv3_count = 0
		GCPv4_count = 0

		with open(txt, "r") as f:
			lines = f.readlines()
			
			#lines_split = lines.split(' ')
			#for splitline in lines_split:

			for line in lines:
				splitline = line.split(' ')
				if splitline[0] == '0':
					wood_count += 1
				if splitline[0] == '1':
					GCP_count += 1
				if splitline[0] == '2':
					GCPv2_count += 1
				if splitline[0] == '3':
					GCPv3_count += 1
				if splitline[0] == '4':
					GCPv4_count += 1

			detectionscount = [wood_count,GCP_count,GCPv2_count,GCPv3_count,GCPv4_count]
			#print(expect)
			#print(detectionscount)
			
			if detectionscount != expect:
				count += 1
				nongood_txts_list.append(txt)
				txt_split = txt.split('/')
				jpg_path = os.path.join(path,(txt_split[6]+'_'+txt_split[7]),txt_split[-1].replace('.txt','.jpg'))
				nongood_jpgs_list.append(jpg_path)
		f.close()

	nongood_txts_list.sort()
	nongood_jpgs_list.sort()

	return(nongood_txts_list,nongood_jpgs_list,count)


all_nongood_txts_list = []
all_nongood_jpgs_list = []

for labelpath in folderlist:
	txtlist = create_txt_list_from_folder(labelpath)
	nongood_txts_list,nongood_jpgs_list,count = analyzelabelpath(labelpath,txtlist)
	all_nongood_txts_list.append(nongood_txts_list)
	all_nongood_jpgs_list.append(nongood_jpgs_list)
	#rint(nongood_txts_list)
	print(count)

all_nongood_txts_list.sort()
all_nongood_jpgs_list.sort()



all_nongood_jpgs_list_combined = []
for listt in all_nongood_jpgs_list:
	for jpg in listt:
		all_nongood_jpgs_list_combined.append(jpg)

all_nongood_txts_list_combined = []
for listt in all_nongood_txts_list:
	for txt in listt:
		all_nongood_txts_list_combined.append(txt)

all_nongood_jpgs_list_combined.sort()
all_nongood_txts_list_combined.sort()


for i in range(30):
	randomnumber = random.randint(0,len(all_nongood_txts_list_combined)-1)
	jpg = all_nongood_jpgs_list_combined[randomnumber]
	txt = all_nongood_txts_list_combined[randomnumber]
	#print(jpg)
	
	shutil.copyfile(jpg,os.path.join(analysepathimages,jpg.split('/')[-1]))
	#print(os.path.join(analysepathimages,jpg.split('/')[-1]))

	#print(txt)
	#print(os.path.join(analysepathlabels,txt.split('/')[-1]))

	with open(txt, 'r') as or_file:
		lines = or_file.readlines()
	or_file.close()

	with open(os.path.join(analysepathlabels,txt.split('/')[-1]), 'w') as file:
		for line in lines:
			line_split = line.split(' ')
			#print(line_split[0]+' '+line_split[1]+' '+line_split[2]+' '+line_split[3]+' '+line_split[4])
			file.write(line_split[0]+' '+line_split[1]+' '+line_split[2]+' '+line_split[3]+' '+line_split[4]+'\n')
	file.close()