import numpy as np
import os
import matplotlib.pyplot as plt

dronenumber = 6
vidnumber = 3

# Define input and output folders
input_folder = 'allparts'

output_folder = input_folder+'_data'

# Ensure output folder exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame1 = 0
#n_frames = 25000
#max_ID = 1052

#find the max amount of frames
numbers = []

for file in os.listdir(input_folder):
    number = int((file.split('frame')[-1]).split('.txt')[0])
    numbers.append(number)

numbers.sort()
#print(numbers)
n_frames = numbers[-1]

for id in range(1, 10000):
    print(id)
    summary = np.zeros((n_frames - frame1 + 1, 4))  # Corrected the array dimensions
    total_diagonal_length = 0
    valid_frame_count = 0
    for i in range(frame1, n_frames + 1):
        name = f'DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber)+'_frame'+str(i).zfill(5)+'.txt'
        filepath = os.path.join(input_folder, name)
        if os.path.isfile(filepath):
            temp = np.loadtxt(filepath)
            if temp.ndim == 1:
                temp = temp.reshape(1, -1)
            found = False
            for j in range(len(temp)):
                if temp[j, 6] == id:
                    summary[i - frame1, :] = temp[j, 1:5]
                    width = temp[j, 3] * 3840  # Multiply width with 3840
                    height = temp[j, 4] * 2160  # Multiply height with 2160
                    diagonal_length = np.sqrt(width**2 + height**2)  # Calculate diagonal length
                    total_diagonal_length += diagonal_length
                    valid_frame_count += 1
                    found = True
                    break
            if not found:
                summary[i - frame1, :] = np.nan
        else:
            summary[i - frame1, :] = np.nan

    # Calculate average diagonal length
    if valid_frame_count > 0:
        average_diagonal = total_diagonal_length / valid_frame_count
    else:
        average_diagonal = np.nan
    print(summary)
    anydata = np.isnan(summary).all()
    print(anydata)
    if anydata:
       max_ID = id - 1
       break
    print(summary)
    name_save = f'summary_ID{id}.npy'
    np.save(os.path.join(output_folder, name_save), summary)
    #print(average_diagonal)
    # Save average diagonal to separate .npy file
    avg_diagonal_save_name = f'average_diagonal_ID{id}.npy'
    np.save(os.path.join(output_folder, avg_diagonal_save_name), average_diagonal)





# Visualizing
plt.figure(1, figsize=(10, 6))

for id in range(1, max_ID + 1):
    name = f'summary_ID{id}.npy'
    summary = np.load(os.path.join(output_folder, name))
    plt.plot(summary[:, 0], summary[:, 1], '.')

plt.savefig(os.path.join(output_folder, 'trajectories.jpg'))
plt.savefig(os.path.join(output_folder, 'trajectories.svg'))
