import os
import shutil

curdir = os.getcwd()

cut_data_folder = os.path.join(curdir,'cut_data')

for i in range(5):
	for dronenumber in [1,6,16]:
		vidnumber = i + 1

		foldername = 'vid'+str(vidnumber)+'_correct_homs'

		fullpath_vid_homs = os.path.join(curdir,foldername)

		to_copy_to_path = os.path.join(cut_data_folder,'DCIM-drone'+str(dronenumber),'drone'+str(dronenumber)+'vid'+str(vidnumber),'correct_homs')
		
		if os.path.exists(to_copy_to_path) == False:
			os.mkdir(to_copy_to_path)

		for file in os.listdir(fullpath_vid_homs):
			if file[-2:] == '.p':
				file_drone_number = int((file.split('DCIM-drone')[-1]).split('_')[0])
				file_vid_number = int((file.split('vid')[-1]).split('_')[0])

				if file_drone_number == dronenumber and file_vid_number == vidnumber:
					outputfilename = (file.split('_DCIM')[-1]).replace('-drone','DCIM-drone')
					outputfilepath = os.path.join(to_copy_to_path,outputfilename)

					shutil.copyfile(os.path.join(fullpath_vid_homs,file), outputfilepath)
					print(os.path.join(fullpath_vid_homs,file))
					print(outputfilepath)
					print(' ')