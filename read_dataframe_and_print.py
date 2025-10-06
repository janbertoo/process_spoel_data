import pickle
import pandas as pd
import numpy as np
from math import atan2, degrees
import ast
from scipy.spatial import KDTree  # For fast nearest-neighbor lookup
import scipy.stats as stats
import matplotlib.pyplot as plt

vidnumbers = [1, 2, 3, 4, 5]

data = []
total_rotation_df = pd.DataFrame(data)

for vidnumber in vidnumbers:
    # Replace 'your_file.pkl' with the path to your pickle file
    with open('step9_dataframe/dataframe_vid' + str(vidnumber) + '_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)

    print(data)

    