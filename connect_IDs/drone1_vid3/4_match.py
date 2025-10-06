import os
import numpy as np
import math

dronenumber = 1
vidnumber = 3

center_threshold= 0.005
diagonal_threshold = 0.5

print('center_threshold')
print(center_threshold)
print('diagonal_threshold')
print(diagonal_threshold)
print('')


def load_trajectory(file_path):
    return np.load(file_path)

def load_average_diagonal(file_path):
    return np.load(file_path)

def load_detections(file_path):
    return np.loadtxt(file_path)


def save_detected(file_path, detections, ID):
    try:
        # Extract numeric part of the ID (remove 'ID' prefix)
        ID_numeric = int(ID[2:])
        
        # Convert detections to NumPy array if it's a list
        if isinstance(detections, list):
            detections = np.array(detections)

        # Ensure detections array is 2D
        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)

        # Create a string representing the detections_with_ID
        detections_with_ID_str = '\n'.join([' '.join([f'{val:.6f}' if idx != 0 else f'{val:.0f}' for idx, val in enumerate(row)]) + f' {ID_numeric}' for row in detections])

        # Write the string to the file
        with open(file_path, 'a') as f:
            f.write(detections_with_ID_str + '\n')
    except Exception as e:
        print(f"Error saving detections to {file_path}: {e}")

        print(file_path)
        print(detections)
        print(ID)
        print('')


def merge_detections(detections):#,X,Y):
    if len(detections) == 1:
        return detections[0]

    #print(X)
    #print(Y)
    #print(detections)
    #print('')

    x_centers = detections[:, 1]
    y_centers = detections[:, 2]
    widths = detections[:, 3]
    heights = detections[:, 4]

    top_left_x = min(x_centers - 0.5 * widths)
    top_left_y = min(y_centers - 0.5 * heights)
    top_right_x = max(x_centers + 0.5 * widths)
    top_right_y = min(y_centers - 0.5 * heights)
    bottom_right_x = max(x_centers + 0.5 * widths)
    bottom_right_y = max(y_centers + 0.5 * heights)
    bottom_left_x = min(x_centers - 0.5 * widths)
    bottom_left_y = max(y_centers + 0.5 * heights)

    merged_x_center = (top_left_x + bottom_right_x) / 2
    merged_y_center = (top_left_y + bottom_right_y) / 2
    merged_width = bottom_right_x - top_left_x
    merged_height = bottom_right_y - top_left_y

    confidence = detections[0, 5]  # assuming confidence is same for all detections

    return [0, merged_x_center, merged_y_center, merged_width, merged_height, confidence]


def load_trajectory(file_path):
    try:
        trajectory = np.load(file_path)
        if len(trajectory.shape) != 2:
            print(f"Error: Invalid shape of trajectory array loaded from {file_path}")
            return None
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory from {file_path}: {e}")
        return None

def load_detections(file_path):
    try:
        detections = np.loadtxt(file_path)
        if detections.ndim == 1:  # If there's only one detection, convert it to a 2D array
            detections = np.expand_dims(detections, axis=0)
        return detections
    except Exception as e:
        print(f"Error loading detections from {file_path}: {e}")
        return None


def main():
    count = 0
    inferred_perdrone_folder = 'inferred_perdrone'
    interpolated_detected_folder = 'interpolated_detected'
    allparts_data_folder = 'allparts_data'

    os.makedirs(interpolated_detected_folder, exist_ok = True)

    for traj_file in os.listdir(allparts_data_folder):
        if traj_file.startswith('akima_trajectory'):
            ID = traj_file.split('_')[-1].split('.')[0]  # Extracting the ID number correctly
            trajectory_file = os.path.join(allparts_data_folder, traj_file)
            trajectory = load_trajectory(trajectory_file)
            
            if trajectory is None or len(trajectory.shape) != 2:
                print(f"Failed to load or invalid trajectory file: {trajectory_file}")
                continue

            average_diagonal_file = os.path.join(allparts_data_folder, f'average_diagonal_{ID}.npy')
            if not os.path.exists(average_diagonal_file):
                print(f"Average diagonal file not found for ID {ID}")
                continue

            average_diagonal = load_average_diagonal(average_diagonal_file)

            for i in range(len(trajectory)):
                frame_name = f"frame{str(i).zfill(5)}"
                detection_file = os.path.join(inferred_perdrone_folder, f'DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber)+'_'+str(frame_name)+'.txt')
                #print(detection_file)
                #print(f"Loading detections from file: {detection_file}")
                if not os.path.exists(detection_file):
                    #print(f"Detections file not found for frame: {frame_name}")
                    continue

                detections = load_detections(detection_file)
                #print("Detections:", detections)

                if detections is None:
                    print(f"Failed to load detections for frame: {frame_name}")
                    continue


                selected_detections = []
                for detection in detections:
                    x_center = detection[1]
                    y_center = detection[2]
                    
                    width = detection[3] * 3840
                    height = detection[4] * 2160
                    diagonal = np.sqrt(width**2 + height**2)
                    
                    #if math.isnan(trajectory[i][0]) == False:
                    #    print(x_center)
                    #    print(trajectory[i][0])
                    #    print(y_center)
                    #    print(trajectory[i][1])
                    #    print(diagonal)
                    #    print(average_diagonal)
                    #    print(' ')

                    if abs(x_center - trajectory[i][0]) <= center_threshold and abs(y_center - trajectory[i][1]) <= center_threshold and abs(diagonal - average_diagonal) <= diagonal_threshold * average_diagonal:
                        #print('hoi')
                        selected_detections.append(detection)

                if selected_detections:
                    if len(selected_detections) == 1:
                        count = count +1
                    merged_detection = merge_detections(np.array(selected_detections))#,trajectory[i][0],trajectory[i][1])
                    interpolated_detected_file = os.path.join(interpolated_detected_folder, f'DCIM-drone'+str(dronenumber)+'_drone'+str(dronenumber)+'vid'+str(vidnumber)+'_'+str(frame_name)+'.txt')
                    #print(interpolated_detected_file)
                    save_detected(interpolated_detected_file, merged_detection, ID)
    print(count)

if __name__ == "__main__":
    main()
