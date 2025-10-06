import os

curdir = os.getcwd()
mainpath = os.path.join(curdir,'cut_data')

vidnumber = 5

vid1drone1 = ['DCIM-drone1_drone1vid'+str(vidnumber)]
vid1drone6 = ['DCIM-drone6_drone6vid'+str(vidnumber)]
vid1drone16 = ['DCIM-drone16_drone16vid'+str(vidnumber)]

drone1frame_drone6frame_drone16frame = [49,0,328]
drone1frame_drone6frame_drone16frame = [60,0,211]
drone1frame_drone6frame_drone16frame = [30,0,115]
drone1frame_drone6frame_drone16frame = [24,0,22]
drone1frame_drone6frame_drone16frame = [0,91,8]

fps = 24
freq = 4

#create list of inference txt files

#create list of homographies

#load inference txt

#load homography data

def getlistofalldroneimages(mainpath,drone_vid_list):
    alldroneimages = []

    for folder in drone_vid_list:
        imagespath = os.path.join(mainpath,folder)
        for imfile in os.listdir(imagespath):
            if imfile[-4:] == '.jpg' and int(imfile[-9:-4]) % (fps/freq) == 0:
                alldroneimages.append(os.path.join(imagespath,imfile))

    alldroneimages.sort()

    return(alldroneimages)

alldrone_1_images = getlistofalldroneimages(mainpath,vid1drone1)
alldrone_6_images =  getlistofalldroneimages(mainpath,vid1drone6)
alldrone_16_images =  getlistofalldroneimages(mainpath,vid1drone16)

stagered_drone_ims = []

for i in range(0,20000):

    if i-drone1frame_drone6frame_drone16frame[0] < 0:
        drone1im = None
    else:
        try:
            drone1im = alldrone_1_images[i-drone1frame_drone6frame_drone16frame[0]]
        except:
            drone1im = None
    #print(drone1im)
    if i-drone1frame_drone6frame_drone16frame[1] < 0:
        drone6im = None
    else:
        try:
            drone6im = alldrone_6_images[i-drone1frame_drone6frame_drone16frame[1]]
        except:
            drone6im = None
    #print(drone6im)
    if i-drone1frame_drone6frame_drone16frame[2] < 0:
        drone16im = None
    else:
        try:
            drone16im = alldrone_16_images[i-drone1frame_drone6frame_drone16frame[2]]
        except:
            drone16im = None
    #print(drone16im)
    if drone1im == None and drone6im == None and drone16im == None:
        break

    stagered_drone_ims.append([drone1im,drone6im,drone16im])

#print(stagered_drone_ims)

stagered_drone_txts = []

stagered_drone_txts_alternative = []

for i in range(len(stagered_drone_ims)):
    try:
        jpg = stagered_drone_ims[i][0]
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps,(jpg_split[-1]).replace('.jpg','.txt'))
        drone1_txt = txt

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps_alternative,(jpg_split[-1]).replace('.jpg','.txt'))
        drone1_txt_alternative = txt
    except:
        drone1_txt = None
        drone1_txt_alternative = None
        
    try:
        jpg = stagered_drone_ims[i][1]
        #print(jpg)
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')
        #print(dcim_split)

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps,(jpg_split[-1]).replace('.jpg','.txt'))
        #print(txt)
        drone6_txt = txt
        
        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps_alternative,(jpg_split[-1]).replace('.jpg','.txt'))
        drone6_txt_alternative = txt
    except:
        drone6_txt = None
        drone6_txt_alternative = None
    
    try:
        jpg = stagered_drone_ims[i][2]
        jpg_split = jpg.split('/')
        dcim = jpg_split[-2]
        dcim_split = dcim.split('_')

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps,(jpg_split[-1]).replace('.jpg','.txt'))
        drone16_txt = txt

        txt = os.path.join(mainpath,dcim_split[0],dcim_split[1],folder_with_gcps_alternative,(jpg_split[-1]).replace('.jpg','.txt'))
        drone16_txt_alternative = txt
    except:
        drone16_txt = None
        drone16_txt_alternative = None
    
    stagered_drone_txts.append([drone1_txt,drone6_txt,drone16_txt])
    stagered_drone_txts_alternative.append([drone1_txt_alternative,drone6_txt_alternative,drone16_txt_alternative])




#print(stagered_drone_ims)
#print(stagered_drone_txts)

names_georeferenced_image = []
countframe = 0
for i in range(len(stagered_drone_ims)):
	name = os.path.join('frame'+str(countframe).zfill(5)+'.jpg')
	countframe += 1
	names_georeferenced_image.append(name)

for i in range(len(stagered_drone_ims)):

	hom_drone1_flag = False
	hom_drone2_flag = False
	hom_drone3_flag = False

	try:
		name_georeferenced_image = names_georeferenced_image[i]
		print(names_georeferenced_image[i])

		hom1 = names_georeferenced_image[i].replace('.jpg','_hom_drone1.p')
		hom2 = names_georeferenced_image[i].replace('.jpg','_hom_drone2.p')
		hom3 = names_georeferenced_image[i].replace('.jpg','_hom_drone3.p')

		
		try:
			name_original_image = stagered_drone_ims[i][0]
			print(name_original_image)
			droneNumberString = (name_original_image.split('drone')[-1]).split('vid')[0]
			vidNumberString = (name_original_image.split('vid')[-1]).split('_')[0]
			frameNumberString = (name_original_image.split('frame')[-1]).split('.jpg')[0]

			homography = name_georeferenced_image.split('.jpg')[0]+'_hom_drone1.p'

			homographyPath = mainpath+'/vid'+str(vidnumber)+'/'+homography
			print(homographyPath)

			IMnameTXT = (name_original_image.split('/')[-1]).replace('.jpg','.txt')
			detections = mainpath+'/DCIM-drone'+droneNumberString+'/drone'+droneNumberString+'vid'+vidNumberString+'/inferred/'+IMnameTXT
			print(detections)
		except:
			print('no drone1 data')

		try:
			name_original_image2 = stagered_drone_ims[i][1]
			print(name_original_image2)
			droneNumberString = (name_original_image2.split('drone')[-1]).split('vid')[0]
			vidNumberString = (name_original_image2.split('vid')[-1]).split('_')[0]
			frameNumberString = (name_original_image2.split('frame')[-1]).split('.jpg')[0]

			homography = name_georeferenced_image.split('.jpg')[0]+'_hom_drone2.p'

			homographyPath = mainpath+'/vid'+str(vidnumber)+'/'+homography
			print(homographyPath)

			IMnameTXT = (name_original_image2.split('/')[-1]).replace('.jpg','.txt')
			detections = mainpath+'/DCIM-drone'+droneNumberString+'/drone'+droneNumberString+'vid'+vidNumberString+'/inferred/'+IMnameTXT
			print(detections)
		except:
			print('no drone6 data')

		try:
			name_original_image3 = stagered_drone_ims[i][2]
			print(name_original_image3)
			droneNumberString = (name_original_image3.split('drone')[-1]).split('vid')[0]
			vidNumberString = (name_original_image3.split('vid')[-1]).split('_')[0]
			frameNumberString = (name_original_image3.split('frame')[-1]).split('.jpg')[0]

			homography = name_georeferenced_image.split('.jpg')[0]+'_hom_drone3.p'

			homographyPath = mainpath+'/vid'+str(vidnumber)+'/'+homography
			print(homographyPath)

			IMnameTXT = (name_original_image3.split('/')[-1]).replace('.jpg','.txt')
			detections = mainpath+'/DCIM-drone'+droneNumberString+'/drone'+droneNumberString+'vid'+vidNumberString+'/inferred/'+IMnameTXT
			print(detections)
		except:
			print('no drone16 data')
		print(' ')
	except:
		print('niet gelukt')

print(mainpath)