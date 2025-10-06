import os
import csv

vidnumber = 5

csv_file = 'vid'+str(vidnumber)+'combined.csv'

# Define folder paths
folder_drone_1 = 'DCIM-drone1/drone1vid'+str(vidnumber)+'/interpolated_detected'
folder_drone_6 = 'DCIM-drone6/drone6vid'+str(vidnumber)+'/interpolated_detected'
folder_drone_16 = 'DCIM-drone16/drone16vid'+str(vidnumber)+'/interpolated_detected'

new_folder_drone_1 = folder_drone_1 + '_linked_IDs'
new_folder_drone_6 = folder_drone_6 + '_linked_IDs'
new_folder_drone_16 = folder_drone_16 + '_linked_IDs'

# Create new folders if they don't exist
os.makedirs(new_folder_drone_1, exist_ok=True)
os.makedirs(new_folder_drone_6, exist_ok=True)
os.makedirs(new_folder_drone_16, exist_ok=True)

# Read CSV file to create separate ID mappings for each drone
id_mapping_drone_1 = {}
id_mapping_drone_6 = {}
id_mapping_drone_16 = {}

with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        final_id, drone16_id, drone6_id, drone1_id = row
        id_mapping_drone_1[drone1_id] = final_id
        id_mapping_drone_6[drone6_id] = final_id
        id_mapping_drone_16[drone16_id] = final_id

# Function to replace IDs in a line
def replace_ids(line, id_mapping):
    parts = line.split()
    drone_id = parts[-1]  # Extract drone ID from last element
    final_id = id_mapping.get(drone_id, None)
    if final_id:
        parts[-1] = final_id  # Replace last element with final ID if found
    return ' '.join(parts)

print(id_mapping_drone_1)
print(id_mapping_drone_6)
print(id_mapping_drone_16)

# Process files in drone1 folder
for filename in os.listdir(folder_drone_1):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_drone_1, filename), 'r') as infile:
            with open(os.path.join(new_folder_drone_1, filename), 'w') as outfile:
                for line in infile:
                    modified_line = replace_ids(line, id_mapping_drone_1)
                    outfile.write(modified_line + '\n')

# Process files in drone6 folder
for filename in os.listdir(folder_drone_6):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_drone_6, filename), 'r') as infile:
            with open(os.path.join(new_folder_drone_6, filename), 'w') as outfile:
                for line in infile:
                    modified_line = replace_ids(line, id_mapping_drone_6)
                    outfile.write(modified_line + '\n')

# Process files in drone16 folder
for filename in os.listdir(folder_drone_16):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_drone_16, filename), 'r') as infile:
            with open(os.path.join(new_folder_drone_16, filename), 'w') as outfile:
                for line in infile:
                    modified_line = replace_ids(line, id_mapping_drone_16)
                    outfile.write(modified_line + '\n')
