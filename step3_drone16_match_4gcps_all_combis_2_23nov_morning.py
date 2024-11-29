import os
import numpy as np
import random

#special coordinates with ['detecting number', index in csv file]
#special_coordinates = [['2',2],['3',3],['5',8],['6',9]]

curdir = os.getcwd()
#print(curdir)
csvdata = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_spoelgcps_notest_adjusted12_plus_rock_bloem.csv')
#csvdata = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_spoelgcps_notest_adjusted12.csv')

#print(csvdata)
csvdataforprocessing = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_spoelgcps_notest_plus_rock_bloem.csv')
#csvdataforprocessing = os.path.join(curdir,'GPS_Data-20230619T123957Z-001/GPS_Data/drone16_spoelgcps_notest.csv')

dronepath = os.path.join(curdir,'cut_data','DCIM-drone16')

def get_all_points_list_from_txt_file(txt_file):
    all_point_list_txt_file = []

    with open(txt_file, "r") as txt_file:
        lines_txt_file = txt_file.readlines()

        GCPv2 = None
        GCPv3 = None
        GCProck = None
        GCPbloem = None

        for i in range(len(lines_txt_file)):
            
            xcoor_txt_file = float(lines_txt_file[i].split(' ')[1])*3840
            ycoor_txt_file = float(lines_txt_file[i].split(' ')[2])*2160

            point_tuple_txt_file = (xcoor_txt_file,ycoor_txt_file)

            #if lines_txt_file[i][0] == '1':
            all_point_list_txt_file.append(point_tuple_txt_file)


            if lines_txt_file[i][0] == '2':
                GCPv2 = [i,1]
            if lines_txt_file[i][0] == '3':
                GCPv3 = [i,2]
            if lines_txt_file[i][0] == '5':
                GCProck = [i,8]
            if lines_txt_file[i][0] == '6':
                GCPbloem = [i,9]

        specialcoords = [GCPv2,GCPv3,GCProck,GCPbloem]
        if GCProck != None:
            print('found GCProck in ',txt_file)
        if GCPbloem != None:
            print('found GCPbloem in ',txt_file)
        #print(specialcoords)

    return(all_point_list_txt_file,specialcoords)

def read_points_into_tuple_3d(csvfile):
    point_list = []

    with open(csvfile, "r") as f:
        lines = f.readlines()

        for line in lines:
            xcoor = float(line.split(',')[1])
            ycoor = float(line.split(',')[2])
            zcoor = float(line.split(',')[3])

            point_tuple = (xcoor,ycoor,zcoor)

            point_list.append(point_tuple)
        
    return(point_list)

def read_points_into_tuple(csvfile):
    point_list = []

    with open(csvfile, "r") as f:
        lines = f.readlines()

        for line in lines:
            xcoor = float(line.split(',')[1])
            ycoor = float(line.split(',')[2])

            point_tuple = (xcoor,ycoor)

            point_list.append(point_tuple)
        
    return(point_list)

def get_ratios(point_list):
    #within this function define 3 ratio's: minmax, middlemax and minmiddle

    distance_squared = {} #create dictionary
    distance_squared_list = []

    for i in range(len(point_list)):
        for j in range(i+1,len(point_list)):
            #if i != j:
            distance_squared[(i,j)] = ( point_list[i][0] - point_list[j][0] ) ** 2 + ( point_list[i][1] - point_list[j][1] ) ** 2
            distance_squared_list.append(( point_list[i][0] - point_list[j][0] ) ** 2 + ( point_list[i][1] - point_list[j][1] ) ** 2)
    ratios = []
    for i in range(len(distance_squared_list)):
        for j in range(i+1,len(distance_squared_list)):
            ratios.append(distance_squared_list[i]/distance_squared_list[j])
            #if distance_squared_list[i] > distance_squared_list[j]:
            #    ratios.append(distance_squared_list[j]/distance_squared_list[i])
            #if distance_squared_list[j] > distance_squared_list[i]:
            #    ratios.append(distance_squared_list[i]/distance_squared_list[j])

    return(ratios)

def get_all_ratios(list_of_ground_control_points):
    r = {}
    for i in range(len(list_of_ground_control_points)):
        for j in range(len(list_of_ground_control_points)):
            if j != i:
                for k in range(len(list_of_ground_control_points)):
                    if k != j and k != i:
                        for l in range(len(list_of_ground_control_points)):
                            if l != k and l != j and l != i:
                                r[(i,j,k,l)] = get_ratios([list_of_ground_control_points[i],list_of_ground_control_points[j],list_of_ground_control_points[k],list_of_ground_control_points[l]])
    return(r)

def find_all_gcps_in_text_file(lookup_table,txt_file):
    point_list_txt_file,specialcoords = get_all_points_list_from_txt_file(txt_file)

    amount_of_gcps = len(point_list_txt_file)


    combinations = []

    for i in range(amount_of_gcps):
        for j in range(i+1,amount_of_gcps):
            for k in range(j+1,amount_of_gcps):
                for l in range(k+1,amount_of_gcps):
                    combinations.append((i,j,k,l))


    

    #print(data_trimmed)
    count = 0
    combinations_and_keys = []
    for combination in combinations:
        points_to_use = [point_list_txt_file[combination[0]],point_list_txt_file[combination[1]],point_list_txt_file[combination[2]],point_list_txt_file[combination[3]]]
        data_txt_file = get_all_ratios(points_to_use)

            
        
        for key in data:
            #print(' ')
            
            skip = False
            for item in specialcoords:
                #print(item)
                if item != None:
                    for i in range(len(key)):
                        if combination[i] == item[0] and key[i] != item[1]:

                            skip = True
                        if key[i] == item[1] and combination[i] != item[0]:


                            skip = True

            
            if skip == True:
                count += 1
                
                continue

            combinations_and_keys.append([combination,key])

    totmindata = []

    for i in range(len(combinations_and_keys)):
        combination = combinations_and_keys[i][0]
        key = combinations_and_keys[i][1]

        points_to_use = [point_list_txt_file[combination[0]],point_list_txt_file[combination[1]],point_list_txt_file[combination[2]],point_list_txt_file[combination[3]]]
        data_txt_file = get_all_ratios(points_to_use)

        totmin = 0
            
        for i in range(len(data[key])):
            totmin = totmin + np.abs(data[key][i] - data_txt_file[(0, 1, 2, 3)][i])

        totmindata.append([totmin,key,combination])

    totmindata.sort()


    combinationshad = []
    
    key_for_gcp = []
    

    for i in range(len(totmindata)):
        combination = totmindata[i][2]
        if combination in combinationshad:
            continue
            
        combinationshad.append(combination)
        key_for_gcp.append(totmindata[i][1])
        if len(combinationshad) == len(combinations):
            break

    gcps_found = []
    keys_found = []
    gcps_found_with_keys = []
    for i in range(len(combinationshad)):
        for j in range(len(combinationshad[i])):
                
            if combinationshad[i][j] in gcps_found:
                continue
            if key_for_gcp[i][j] in keys_found:
                continue
                
            gcps_found.append(combinationshad[i][j])
            keys_found.append(key_for_gcp[i][j])
            
            gcps_found_with_keys.append([combinationshad[i][j],key_for_gcp[i][j]])
       
    return(gcps_found_with_keys,point_list_txt_file)


if __name__ == '__main__':
    point_list = read_points_into_tuple(csvdata)
    actual_point_list = read_points_into_tuple_3d(csvdataforprocessing)

    data = get_all_ratios(point_list)

    pathsdrone = []


    for item in os.listdir(dronepath):
        pathsdrone.append(os.path.join(dronepath,item))

    for basepath in pathsdrone:

        path = os.path.join(basepath,'labels')
        outputpath = path.replace('labels','labels_with_gcps')
        
        if os.path.isdir(outputpath) == False:
            os.mkdir(outputpath)

        for file in os.listdir(path):
            if file[-4:] == '.txt':
                txt_file = os.path.join(path,file)
            #print(' ')
            #print(txt_file)
            outputtxtpath = txt_file.replace('labels','labels_with_gcps')

            if os.path.exists(outputtxtpath) == False or os.path.exists(outputtxtpath) == True:
                gcps_found_with_keys,point_list_txt_file = find_all_gcps_in_text_file(data,txt_file)
                print(gcps_found_with_keys)
                
                with open(outputtxtpath, 'w') as outfile:
                    for i in range(len(gcps_found_with_keys)):

                        line_to_write = str(point_list_txt_file[gcps_found_with_keys[i][0]][0])+' '+str(point_list_txt_file[gcps_found_with_keys[i][0]][1])+', '+str(actual_point_list[gcps_found_with_keys[i][1]][0])+' '+str(actual_point_list[gcps_found_with_keys[i][1]][1])+' '+str(actual_point_list[gcps_found_with_keys[i][1]][2])+'\n'
                        outfile.write(line_to_write)
