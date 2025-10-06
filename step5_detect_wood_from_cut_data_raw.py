import cv2
import os
import datetime
from ultralytics import YOLO

curdir = os.getcwd()

des_folder = os.path.join(curdir,'cut_data')

model1_path = os.path.join(curdir,'Train-YOLOv8-on-Custom-Dataset-A-Complete-Tutorial/wood_detection_trained_13dec_best.pt')

model = YOLO(model1_path)
#model.to('cuda')

#drone_name = 'gekkedrone'
#vid_name = 'gekkevid'


def infer(image_path,drone_name,vid_name,des_folder,model):

	results = model.predict(source=image_path, save=False, save_txt=True, save_conf=True, project=drone_name, name=vid_name, exist_ok=True, conf=0.05)



cutpaths = [
	os.path.join(curdir,'cut_data/DCIM-drone1_drone1vid1'),
	os.path.join(curdir,'cut_data/DCIM-drone1_drone1vid2'),
	os.path.join(curdir,'cut_data/DCIM-drone1_drone1vid3'),
	os.path.join(curdir,'cut_data/DCIM-drone1_drone1vid4'),
	os.path.join(curdir,'cut_data/DCIM-drone1_drone1vid5'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid1'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid2'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid3'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid4'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid5'),
	os.path.join(curdir,'cut_data/DCIM-drone16_drone16vid1'),
	os.path.join(curdir,'cut_data/DCIM-drone16_drone16vid2'),
	os.path.join(curdir,'cut_data/DCIM-drone16_drone16vid3'),
	os.path.join(curdir,'cut_data/DCIM-drone16_drone16vid4'),
	os.path.join(curdir,'cut_data/DCIM-drone16_drone16vid5')
]

for cutpath in cutpaths:
	drone_name = cutpath.split('_')[-1]+'_wood_detections'
	vid_name = (cutpath.split('_')[-2]).split('-')[-1]
	print(drone_name)
	print(vid_name)
	for file in os.listdir(cutpath):
		if file[-4:] == '.jpg':
			imgpath = os.path.join(cutpath,file)
			infer(imgpath,drone_name,vid_name,des_folder,model)
