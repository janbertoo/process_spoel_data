import cv2
import os
import datetime
from ultralytics import YOLO

curdir = os.getcwd()

des_folder = os.path.join(curdir,'cut_data')

model1_path = os.path.join(curdir,'Train-YOLOv8-on-Custom-Dataset-A-Complete-Tutorial/GCP_drone6_bloem2.pt')

model = YOLO(model1_path)

drone_name = 'gekkedrone'
vid_name = 'gekkevid'


def infer(image_path,drone_name,vid_name,des_folder,model):

	results = model.predict(source=image_path, save=False, save_txt=True, save_conf=True, project=drone_name, name=vid_name, exist_ok=True, conf=0.1)
	
	#txtfilepath = os.path.join(des_folder,drone_name,vid_name,'labels',(image_path.split('/')[-1]).replace('.jpg','.txt'))


cutpaths = [
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid1'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid2'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid3'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid4'),
	os.path.join(curdir,'cut_data/DCIM-drone6_drone6vid5')
]

for cutpath in cutpaths:
	for file in os.listdir(cutpath):
		if file[-4:] == '.jpg':
			imgpath = os.path.join(cutpath,file)
			infer(imgpath,drone_name,vid_name,des_folder,model)
