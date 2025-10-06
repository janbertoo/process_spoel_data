import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
import ast  # To safely evaluate the string representation of lists


vidnumbers = [1,2,3,4,5]


for vidnumber in vidnumbers:
    # Replace 'your_file.pkl' with the path to your pickle file
    with open('step9_dataframe/dataframe_vid'+str(vidnumber)+'_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    # Print the contents of the pickle file
    #print(data)






    # Function to calculate the angle between two points
    def calculate_angle(top_left, bottom_right):
        dx = bottom_right[0] - top_left[0]
        dy = bottom_right[1] - top_left[1]
        return degrees(atan2(dy, dx))

    # Function to calculate the midpoint between two points
    def calculate_midpoint(point1, point2):
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        return [mid_x, mid_y]

    # Function to parse strings into lists of floats
    def parse_coordinates(coord_str):
        try:
            # Use ast.literal_eval to safely parse the string representation of the list
            return ast.literal_eval(coord_str)
        except (ValueError, SyntaxError):
            return None

    # Assuming your dataframe is called df
    def create_rotation_df(df):
        # Parse the coordinate columns to ensure they are lists
        df['top_left_geo'] = df['top_left_geo'].apply(parse_coordinates)
        df['bottom_right_geo'] = df['bottom_right_geo'].apply(parse_coordinates)
        df['center_geo'] = df['center_geo'].apply(parse_coordinates)

        # Convert the 'length' column to numeric values (floats)
        df['length'] = pd.to_numeric(df['length'], errors='coerce')

        # Sort the dataframe by 'ID' and 'timestamp'
        df = df.sort_values(by=['ID', 'timestamp'])

        # Create a list to store results
        rotation_data = []

        # Iterate over the dataframe to find pairs of rows with the same 'ID' and timestamp difference of 0.25
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # Check if the ID matches and timestamp difference is 0.25
            if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) == 0.25:
                # Ensure the coordinates are parsed correctly
                if current_row['top_left_geo'] and current_row['bottom_right_geo'] and next_row['top_left_geo'] and next_row['bottom_right_geo']:
                    # Calculate the length difference as a percentage
                    if pd.notnull(current_row['length']) and pd.notnull(next_row['length']):
                        length_diff_percentage = abs(current_row['length'] - next_row['length']) / current_row['length']

                        # Only proceed if the length difference is less than 10%
                        if length_diff_percentage < 0.15:
                            # Calculate the rotation (angle difference between the two rows)
                            angle1 = calculate_angle(current_row['top_left_geo'], current_row['bottom_right_geo'])
                            angle2 = calculate_angle(next_row['top_left_geo'], next_row['bottom_right_geo'])
                            rotation = angle2 - angle1

                            # Calculate the midpoint of the 'center_geo' coordinates
                            if current_row['center_geo'] and next_row['center_geo']:
                                center_between_geo = calculate_midpoint(current_row['center_geo'], next_row['center_geo'])

                                # Append the result to the list
                                rotation_data.append({
                                    'ID': current_row['ID'],
                                    'timestamp_1': current_row['timestamp'],
                                    'timestamp_2': next_row['timestamp'],
                                    'rotation': rotation,
                                    'center_between_geo': center_between_geo
                                })

        # Create a new dataframe from the collected rotation data
        rotation_df = pd.DataFrame(rotation_data)
        return rotation_df

    # Example usage with your dataframe
    rotation_df = create_rotation_df(data)

    # Print the resulting dataframe
    print(rotation_df)

