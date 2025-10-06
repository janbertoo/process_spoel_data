import os
import shutil

curdir = os.getcwd()

frameinterval = 10

#vidnumber = 1

for i in range(5):
	vidnumber = i + 1

	in_file = os.path.join(curdir,'vid'+str(vidnumber))
	out_file = os.path.join(curdir,'vid'+str(vidnumber)+'_correct_homs')

	frameslists = []

	for i in range(len(os.listdir(in_file))):
		framename = 'frame'+str(i).zfill(5)
		if i % frameinterval == 0:
			frameslists.append(framename)

	for framename in frameslists:
		#print(framename)
		for file in os.listdir(in_file):
			if file[0:10] == framename:
				print(file)
				shutil.copyfile(os.path.join(in_file,file), os.path.join(out_file,file))