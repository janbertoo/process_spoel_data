# process_spoel_data

The processing of the data took a year. A lot of different steps have been performed and manual interventions have had to be done along the way. I have attempted to upload all relevant scripts but chances are there are pieces of the puzzle missing. Please feel free to contact me if you have questions.

The process consisted roughly of the following steps. The most important files are the pickled dataframe files in the 'step9_dataframe' folder. Those files are necessary to run the analysis scripts.

The raw drone video's are a couple of hundred of GB in size and are therefore not uploaded to zenodo. Feel free to contact me to get the data. For verification purposes, the georectified videos with traced instream wood pieces have been uploaded to zenodo.

Step 1: Cut frames and detect GCPS

Step 2: Check GCP detections

Step 3: Match all detected GCPS to real-world GCPS

Step 4: Calculate homographies between pixel coordinates and real-world coordinates

Step 5: Detect wood from videos

Step 6: Georectify the detections

Step 7: Cut the bounding boxes and analyze orientation

Step 8: Georectify trajectories (for visual purposes)

Step 9: Analyze data

Step 10: Analyze angles and rotations
