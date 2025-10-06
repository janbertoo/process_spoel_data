import os
import math
import csv
import pandas as pd

csv_file_path = 'drone16_vid1.csv'
dronenumber = 16
vidnumber = 1
duration = 17345



num_pieces = 10
frames_per_video = duration / num_pieces

curdir = os.getcwd()

def create_frame_mapping(num_pieces, frames_per_video):
    frame_mapping = []

    for i in range(num_pieces):
        section_frames = []
        for j in range(math.ceil(frames_per_video)):
            frame_number = math.floor(i * frames_per_video) + j
            section_frames.append([i + 1, j + 1, frame_number + 1])  # Adding 1 to make frame numbers 1-indexed
        frame_mapping.append(section_frames)

    return frame_mapping

def find_number_in_part(csv_file, part_number, target_number):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        # Iterate through each row in the CSV file
        for row in reader:
            part_column = f'part{part_number}'
            
            # Check if the target_number is in the specified part column
            numbers_in_part = row[part_column].split()
            if target_number in map(int, numbers_in_part):
                return row['Number']
    
    # Return None if the number is not found in the specified part
    return None

frame_mapping = create_frame_mapping(num_pieces, frames_per_video)

alldir = os.path.join(curdir,'allparts')
os.mkdir(alldir)

for folder in os.listdir(curdir):
    folder_splitted = folder.split('_')[-1]
    
    if folder_splitted[0:4] == 'part':
        print(folder)
        partnumber = int(folder_splitted.split('part')[-1])
        #print(partnumber)

        
        
        labelspath = os.path.join(curdir,folder,'labels')
        for file in os.listdir(labelspath):
            print(file)
            part_frame_number = int((file.split('.txt')[0]).split('_')[-1])

            for entry in frame_mapping:
                for framenumbers in entry:
                    if framenumbers[0] == partnumber and framenumbers[1] == part_frame_number:
                        newframenumber = framenumbers[2]

            newfilepath = os.path.join(alldir,'DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber)+'_frame'+str(newframenumber).zfill(5)+'.txt')





            filepath = os.path.join(labelspath,file)
            with open(filepath, 'r') as file:
                # Read all lines into a list
                lines = file.readlines()

            lines_to_be_stored = []

            for line in lines:
                line_splitted = line.split(' ')
                if len(line_splitted) == 7:
                    oldnumber = int(line_splitted[6])
                    newnumber = find_number_in_part(csv_file_path,partnumber,oldnumber)

                    if newnumber != None:
                        newline = line_splitted[0]+' '+line_splitted[1]+' '+line_splitted[2]+' '+line_splitted[3]+' '+line_splitted[4]+' '+line_splitted[5]+' '+str(newnumber)+'\n'
                        with open(newfilepath, 'a') as file:
                            file.write(newline)




def merge_duplicate_rows(file_path):
    # Read the text file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=' ', header=None)

    # Identify and merge duplicate rows based on the 7th column
    df_duplicates = df[df.duplicated(subset=6, keep=False)]

    if not df_duplicates.empty:
        # Create a new DataFrame for merged rows
        merged_df = df_duplicates.groupby(6, as_index=False).agg({1: 'mean', 2: 'mean', 3: 'mean', 4: 'mean', 5: 'mean'})

        # Update the first column to be 0
        merged_df[0] = 0

        # Concatenate the merged DataFrame with non-duplicate rows
        result_df = pd.concat([df[~df.duplicated(subset=6, keep=False)], merged_df], ignore_index=True)

        # Save the result to a new text file
        result_df.to_csv(file_path, sep=' ', header=False, index=False)

# Provide the path to your folder containing .txt files
folder_path = alldir

# Iterate through all .txt files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        merge_duplicate_rows(file_path)



# List all files in the folder
files = os.listdir(alldir)
files.sort()

for file in files:
    if file.endswith(".txt"):
        filenumber_string = file.split('_')[-1]
        # Extract the numeric part from the file name
        file_number = int(filenumber_string[5:-4])  # Assuming the format is "fileXXXXX.txt"
        
        # Subtract 1 from the file number
        new_file_number = file_number - 1
        
        # Create the new file name
        new_file_name = f"DCIM-drone"+str(dronenumber)+"_drone"+str(dronenumber)+"vid"+str(vidnumber)+"_frame"+str(new_file_number).zfill(5)+".txt"
        
        # Rename the file
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
