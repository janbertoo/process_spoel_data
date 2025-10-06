import cv2
import os
import datetime
from ultralytics import YOLO

curdir = os.getcwd()

width = 3840
height = 2160

frequency = 4 #Hz
vidfreq = 24 #Hz (vidfreq/frequency)
timedifferencedroneandreal = 2 #hours

des_folder = os.path.join(curdir,'cut_data')
print(des_folder)
if os.path.isdir(des_folder) == False:
	os.mkdir(des_folder)

model1_path = os.path.join(curdir,'Train-YOLOv8-on-Custom-Dataset-A-Complete-Tutorial/runs/detect/yolov8s_27okt_retrain_2425okt_gcps2/weights/best.pt')

#retrained weights
model2_path = os.path.join(curdir,'Train-YOLOv8-on-Custom-Dataset-A-Complete-Tutorial/runs/detect/drone1_gcps_v8_smaller/weights/best.pt')

folders = [
	os.path.join(curdir,'DCIM-drone6/100MEDIA')
]


def create_list_of_vid_files(DCIMpath):
	vidlist = []
	for file in os.listdir(DCIMpath):
		if file[-4:] == '.MOV' or file[-4:] == '.MP4':
			vidlist.append(os.path.join(DCIMpath,file))
	vidlist.sort()

	return vidlist



def read_vid_create_folder_and_cut_frames(input_vid,des_folder,folder_to_create):#whichdrone,whichvid):
	#read vid file to get the start time of the video
	vid_capture = cv2.VideoCapture(input_vid)
	fps = vid_capture.get(cv2.CAP_PROP_FPS) 
	length = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	
	#whichdrone = input_vid.split('/')[-3]
	#whichvid = (input_vid.split('/')[-1]).split('.')[0]
	#print(whichdrone+'_'+whichvid)
	#folder_to_create = os.path.join(des_folder,whichdrone+'_'+whichvid)

	#create the folder if not exists
	if os.path.isdir(folder_to_create) != True:
		os.mkdir(folder_to_create)
		print('create '+str(folder_to_create))
	
	#read vid
	vid_capture = cv2.VideoCapture(input_vid)

	count = 0
	#cut frames and save to folder
	while(vid_capture.isOpened()):
		ret, frame = vid_capture.read()
		if ret == False:
			print('break')
			break
		
		if count % (vidfreq/frequency) == 0:
			cv2.imwrite(os.path.join(folder_to_create,(whichdrone+'_'+whichvid+'_frame'+str("{:05d}".format(count))+'.jpg')),frame)
		
		count += 1
	
	vid_capture.release()

def infer(image_path,drone_name,vid_name,des_folder,model):
	results = model.predict(source=image_path, save=False, save_txt=True, save_conf=True, project=drone_name, name=vid_name, exist_ok=True)
	txtfilepath = os.path.join(des_folder,drone_name,vid_name,'labels',(image_path.split('/')[-1]).replace('.jpg','.txt'))

	try:
		with open(txtfilepath, "r") as f:
				lines = f.readlines()
				center_and_confs = []
				for line in lines:
					splitted_line = line.split(' ')
					center_and_confs.append([float(splitted_line[1]),float(splitted_line[2]),float(splitted_line[5].split('\n')[0])  ] )
		#print('1')
		#print(center_and_confs)
		#center_and_confs = sorted(center_and_confs, key=itemgetter(2))
		#print('3')
		#center_and_confs.reverse()
		#print('2')
		with open(txtfilepath, "w") as f:
				num_of_detects = 0
				for i in range(len(lines)):
					splitted = lines[i].split(' ')
					keep = True
					#llononeside = False
					for n in range(len(center_and_confs)):
						if i != n:
							#print(i,n)
							#print(splitted[1], center_and_confs[n][0])
							#print(abs((float(splitted[1]) * width) - (float(center_and_confs[n][0]) * width)))
							#print(abs((float(splitted[2]) * width) - (float(center_and_confs[n][1]) * width)))
							if abs((float(splitted[1]) * width) - (float(center_and_confs[n][0]) * width)) < 70 and abs((float(splitted[2]) * width) - (float(center_and_confs[n][1]) * width)) < 70:
								#print('DOUBLE')
								if float(splitted[5]) < float(center_and_confs[n][2]):
									keep = False

					if keep == True:
						f.write(lines[i])
						num_of_detects += 1
					#print('keep: ',keep)
					#print(' ')

		if num_of_detects < 4:
			foundenough = False
			print('DID NOT FIND ENOUGH!!!!!!')
		elif num_of_detects == 4:
			#print(float(center_and_confs[0][0]))
			#print(float(center_and_confs[1][0]))
			#print(float(center_and_confs[1][1]))
			if (center_and_confs[0][0]) > 0.5 and (center_and_confs[1][0]) > 0.5 and (center_and_confs[2][0]) > 0.5 and (center_and_confs[3][0]) > 0.5:
				foundenough = False # all on one side
			elif (center_and_confs[0][1]) > 0.5 and (center_and_confs[1][1]) > 0.5 and (center_and_confs[2][1]) > 0.5 and (center_and_confs[3][1]) > 0.5:
				foundenough = False # all on one side
			elif (center_and_confs[0][0]) < 0.5 and (center_and_confs[1][0]) < 0.5 and (center_and_confs[2][0]) < 0.5 and (center_and_confs[3][0]) < 0.5:
				foundenough = False # all on one side
			elif (center_and_confs[0][1]) < 0.5 and (center_and_confs[1][1]) < 0.5 and (center_and_confs[2][1]) < 0.5 and (center_and_confs[3][1]) < 0.5:
				foundenough = False # all on one side
			else:
				foundenough = True
		else:
			foundenough = True

		print('found enough: ',foundenough)

	except:
		print('ERROR')
		foundenough = False

	return foundenough, txtfilepath

for folder in folders:
	print(folder)
	vidlist = create_list_of_vid_files(folder)
	#print(vidlist)
	vidlist.sort()

	for inputvidpath in vidlist:

		whichdrone = inputvidpath.split('/')[-3]
		whichvid = (inputvidpath.split('/')[-1]).split('.')[0]
		print(whichdrone+'_'+whichvid)
		folder_to_create = os.path.join(des_folder,whichdrone+'_'+whichvid)

		read_vid_create_folder_and_cut_frames(inputvidpath,des_folder,folder_to_create)

		os.chdir(des_folder)

		jpg_list = []
		#create list of images
		for file in os.listdir(folder_to_create):
			if file[-4:] == '.jpg':
				jpg_list.append(os.path.join(folder_to_create,file))

		jpg_list.sort()

		model1 = YOLO(model1_path)
		model2 = YOLO(model2_path)

		for jpg in jpg_list:
			#if foundenough == False:
			#	foundenough = infer(jpg,whichdrone,whichvid,des_folder)

			if int((jpg.split('.jpg')[0]).split('_frame')[-1]) % ( vidfreq / frequency ) == 0:# or foundenough == False:
				
				foundenough,txtpath = infer(jpg,whichdrone,whichvid,des_folder,model1)
				#print(foundenough)
				if foundenough == False:
					#remove the txt file
					print('REMOVING txt file because didnt find enough gcps')
					os.remove(txtpath)
					
					foundenough,txtpath = infer(jpg,whichdrone,whichvid,des_folder,model2)
			else:
				os.remove(jpg)