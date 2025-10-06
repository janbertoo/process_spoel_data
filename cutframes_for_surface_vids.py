import cv2
import os
import datetime

curdir = os.getcwd()

width = 3840
height = 2160

frequency = 4 #Hz
vidfreq = 24 #Hz
timedifferencedroneandreal = 2 #hours

des_folder = os.path.join('/home/jean-pierre/Desktop/13feb','cut_data')
print(des_folder)
if os.path.isdir(des_folder) == False:
	os.mkdir(des_folder)

def create_list_of_vid_files(DCIMpath):
	vidlist = []
	for file in os.listdir(DCIMpath):
		if file[-4:] == '.MOV' or file[-4:] == '.MP4':
			vidlist.append(os.path.join(DCIMpath,file))
	vidlist.sort()

	return vidlist

def read_vid_create_folder_and_cut_frames(input_vid,des_folder,folder_to_create,timestart):#whichdrone,whichvid):
	
	timestart = timestart * 24
	#read vid file to get the start time of the video
	vid_capture = cv2.VideoCapture(input_vid)
	fps = vid_capture.get(cv2.CAP_PROP_FPS) 
	length = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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

		#if count % (vidfreq/frequency) == 0:
		if timestart < count < timestart + 60:
			print(os.path.join(folder_to_create,(whichdrone+'_'+whichvid+'_frame'+str("{:05d}".format(count))+'.jpg')))
			cv2.imwrite(os.path.join(folder_to_create,(whichdrone+'_'+whichvid+'_frame'+str("{:05d}".format(count))+'.jpg')),frame)

		if count > timestart + 61:
			break
		
		count += 1
	
	vid_capture.release()


alldata_dronenumber_etc = [
	#[1,1,417],
	#[1,2,60],
	#[1,3,870],
	#[1,4,200],
	#[1,5,100],
	#[6,1,417],
	#[6,2,60],
	#[6,3,870],
	#[6,4,200],
	#[6,5,70],
	#[16,1,417],
	#[16,2,30],
	#[16,3,870],
	#[16,4,200],
	[16,5,85],
]

for dronenumber_etc in alldata_dronenumber_etc:
	dronenumber = dronenumber_etc[0]
	vidnumber = dronenumber_etc[1]
	timestart = dronenumber_etc[2]

	folder = os.path.join(curdir,'DCIM-drone'+str(dronenumber)+'/100MEDIA')
	#vidlist = []
	for vidfile in os.listdir(folder):
		if vidfile[0:10] == ('drone'+str(dronenumber)+'vid'+str(vidnumber)) or vidfile[0:11] == ('drone'+str(dronenumber)+'vid'+str(vidnumber)):
			inputvidpath = os.path.join(folder,vidfile)
			print(inputvidpath)
	#print(folder)
	#vidlist = create_list_of_vid_files(folder)
	#print(vidlist)
	#vidlist.sort()

	#for inputvidpath in vidlist:

	whichdrone = inputvidpath.split('/')[-3]
	whichvid = (inputvidpath.split('/')[-1]).split('.')[0]
	print(whichdrone+'_'+whichvid)
	folder_to_create = os.path.join(des_folder,whichdrone+'_'+whichvid)

	read_vid_create_folder_and_cut_frames(inputvidpath,des_folder,folder_to_create,timestart)

