import pickle
import pandas as pd

vidnumbers = [1, 2, 3, 4, 5]

# Initialize a dictionary to store total volumes per video
total_volumes_per_video = {}

for vidnumber in vidnumbers:
    # Load the data for each video
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data)
    
    # Group by 'ID' to ensure each ID is used only once
    unique_logs = df.groupby('ID').first()
    
    # Sum the 'volume_median' for each unique ID
    total_volume = unique_logs['volume_median'].sum()
    
    # Store the total volume for this video
    total_volumes_per_video[vidnumber] = total_volume

    print(f"Total wood volume for video {vidnumber}: {total_volume:.3f} cubic meters")

# Display the total volumes for all videos
print("Total wood volumes per video:")
for vid, volume in total_volumes_per_video.items():
    print(f"Video {vid}: {volume:.3f} cubic meters")

