import os
import numpy as np

x=3840
y=2160

curdir = os.getcwd()
#print(curdir)
csvdata = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_spoelgcps_notest_adjusted12.csv')
#print(csvdata)
csvdataforprocessing = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_spoelgcps_notest.csv')

csvdataspecialcoords = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_only_gcp2_gcp3_rock_bloem.csv')


dronepath = os.path.join(curdir,'cut_data','DCIM-drone16')


if __name__ == '__main__':
    
    #coords_special_gcps = []
    with open(csvdataspecialcoords, "r") as special_coords_file:
        special_coords_lines = special_coords_file.readlines()

    print(special_coords_lines[0])#gcp2
    print(special_coords_lines[1])#gcp3
    print(special_coords_lines[2])#gcprock
    print(special_coords_lines[3])#gcpbloem

    pathsdrone = []

    for item in os.listdir(dronepath):
        pathsdrone.append(os.path.join(dronepath,item))

    print(pathsdrone)

    for basepath in pathsdrone:

        path = os.path.join(basepath,'labels')
        outputpath = path.replace('labels','labels_with_gcps')
        
        if os.path.isdir(outputpath) == False:
            os.mkdir(outputpath)

        for file in os.listdir(path):
            print(' ')
            if file[-4:] == '.txt':
                txt_file = os.path.join(path,file)
            
            outputtxtpath = txt_file.replace('labels','labels_with_gcps')

            if os.path.exists(outputtxtpath) == False or os.path.exists(outputtxtpath) == True:

                print(txt_file)
                with open(txt_file, "r") as txt_file:
                    lines_txt_file = txt_file.readlines()


                lines_to_write = []

                for i in range(len(lines_txt_file)):
                    print(lines_txt_file[i])
                    splitted = lines_txt_file[i].split(' ')

                    if splitted[0] == '2': #gcp2
                        splitted_special_coor = special_coords_lines[0].split(',')
                        lines_to_write.append(str(float(splitted[1])*x)+' '+str(float(splitted[2])*y)+', '+splitted_special_coor[1]+' '+splitted_special_coor[2]+' '+splitted_special_coor[3])

                    if splitted[0] == '3': #gcp3
                        splitted_special_coor = special_coords_lines[1].split(',')
                        lines_to_write.append(str(float(splitted[1])*x)+' '+str(float(splitted[2])*y)+', '+splitted_special_coor[1]+' '+splitted_special_coor[2]+' '+splitted_special_coor[3])
                    
                    if splitted[0] == '5': #gcprock
                        splitted_special_coor = special_coords_lines[2].split(',')
                        lines_to_write.append(str(float(splitted[1])*x)+' '+str(float(splitted[2])*y)+', '+splitted_special_coor[1]+' '+splitted_special_coor[2]+' '+splitted_special_coor[3])
                    
                    if splitted[0] == '6': #gcpbloem
                        splitted_special_coor = special_coords_lines[3].split(',')
                        lines_to_write.append(str(float(splitted[1])*x)+' '+str(float(splitted[2])*y)+', '+splitted_special_coor[1]+' '+splitted_special_coor[2]+' '+splitted_special_coor[3])

                print(lines_to_write)
                with open(outputtxtpath, 'w+') as outfile:
                    for i in range(len(lines_to_write)):
                        outfile.write(lines_to_write[i])