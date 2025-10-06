import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from tabulate import tabulate

savefigfolder = 'step9_dataframe/'

vidnumbers = [1,2,3,4,5]

def combine_dataframes(savefigfolder, vidnumbers):
    dataframes = []

    for vidnumber in vidnumbers:
        # Construct file path
        file_path = savefigfolder + 'dataframe_vid' + str(vidnumber) + '.p'
        
        # Read the dataframe
        df = pd.read_pickle(file_path)
        
        # Add a new column with the vidnumber
        df['vidnumber'] = vidnumber
        
        # Add the dataframe to the list
        dataframes.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

df = combine_dataframes(savefigfolder,vidnumbers)

print(df)


# Create a new dataframe with selected columns
selected_columns_df = df[['ID','vidnumber', 'length', 'diameter']]

selected_columns_df_median = df[['ID','vidnumber', 'length_median', 'diameter_median']]

#selected_columns_df['vidnumber_ID'] = str(selected_columns_df['vidnumber']) + '_' + str(selected_columns_df['ID'])
selected_columns_df['vidnumber_ID'] = selected_columns_df['vidnumber'].astype(str) + '_' + selected_columns_df['ID'].astype(str)

# Group by 'ID' and calculate the median for each group
median_values_df = selected_columns_df.groupby('vidnumber_ID').median()
median_values_df['volume'] = median_values_df['length'] * np.pi * (median_values_df['diameter']/2) ** 2

# Convert ID index to numerical values
median_values_df.index = median_values_df.index.astype(int)

# Sort the dataframe by the ID index
median_values_df = median_values_df.sort_index()

#reindex
median_values_df = median_values_df.reset_index(drop=True)

################################################################################# PLOT LENGTHS AND DIAMETERS OF INDIVIDUAL PIECES
# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(20, 10))

# Plot length_median
axes[0].bar(median_values_df.index, median_values_df['length'], color='tab:blue')
axes[0].set_title('Median Length')
axes[0].set_ylabel('Length')
axes[0].axhline(y=1, color='red', linestyle='--')  # Add horizontal line at 1m

# Plot diameter_median
axes[1].bar(median_values_df.index, median_values_df['diameter'], color='tab:orange')
axes[1].set_title('Median Diameter')
axes[1].set_ylabel('Diameter')
axes[1].axhline(y=0.1, color='red', linestyle='--')  # Add horizontal line at 0.1m

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(savefigfolder+'lengths_and_diameters_vidALL.png')
plt.savefig(savefigfolder+'lengths_and_diameters_vidALL.pdf')

# Define bin edges
length_bins = np.arange(0, median_values_df['length'].max() + 0.25, 0.25)
diameter_bins = np.arange(0, median_values_df['diameter'].max() + 0.025, 0.025)

################################################################################# PLOT LENGTHS AND DIAMETERS IN BINS

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot length_median histogram
axes[0].hist(median_values_df['length'], bins=length_bins, color='tab:blue')
axes[0].set_title('Median Length Histogram')
axes[0].set_ylabel('Frequency')
axes[0].set_xlabel('Length (m)')
axes[0].axvline(x=1, color='red', linestyle='--')  # Add vertical line at 1m

# Plot diameter_median histogram
axes[1].hist(median_values_df['diameter'], bins=diameter_bins, color='tab:orange')
axes[1].set_title('Median Diameter Histogram')
axes[1].set_ylabel('Frequency')
axes[1].set_xlabel('Diameter (m)')
axes[1].axvline(x=0.1, color='red', linestyle='--')  # Add vertical line at 0.1m

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(savefigfolder+'lengths_and_diameters_bins_vidALL.png')
plt.savefig(savefigfolder+'lengths_and_diameters_bins_vidALL.pdf')

print('LENGTH')
print(len(median_values_df))
print(median_values_df)
print('')

#median_values_df
smaller_df = median_values_df[median_values_df['diameter'] > 0.10]

large_wood_df = smaller_df[smaller_df['length'] > 1]

large_wood_df = large_wood_df.reset_index(drop=True)
large_wood_df['volume'] = large_wood_df['length'] * np.pi * ( large_wood_df['diameter'] / 2 ) ** 2

print('Large: '+str(len(large_wood_df)))



smaller_df = median_values_df[median_values_df['diameter'] <= 0.10]

semilarge_wood_df_long = smaller_df[smaller_df['length'] > 1]

semilarge_wood_df_long = semilarge_wood_df_long.reset_index(drop=True)
semilarge_wood_df_long['volume'] = semilarge_wood_df_long['length'] * np.pi * ( semilarge_wood_df_long['diameter'] / 2 ) ** 2

print('Long: '+str(len(semilarge_wood_df_long)))


smaller_df = median_values_df[median_values_df['diameter'] > 0.10]

semilarge_wood_df_thick = smaller_df[smaller_df['length'] <= 1]

semilarge_wood_df_thick = semilarge_wood_df_thick.reset_index(drop=True)
semilarge_wood_df_thick['volume'] = semilarge_wood_df_thick['length'] * np.pi * ( semilarge_wood_df_thick['diameter'] / 2 ) ** 2

print('Thick: '+str(len(semilarge_wood_df_thick)))


smaller_df = median_values_df[median_values_df['diameter'] <= 0.10]

small_wood_df = smaller_df[smaller_df['length'] <= 1]

small_wood_df = small_wood_df.reset_index(drop=True)
small_wood_df['volume'] = small_wood_df['length'] * np.pi * ( small_wood_df['diameter'] / 2 ) ** 2

print('Small: '+str(len(small_wood_df)))


large_long_thick_small = [
    ['Large',len(large_wood_df),'> 1','> 0.1',large_wood_df['volume'].sum()],
    ['Long',len(semilarge_wood_df_long),'> 1','<= 0.1',semilarge_wood_df_long['volume'].sum()],
    ['Thick',len(semilarge_wood_df_thick),'<= 1','> 0.1',semilarge_wood_df_thick['volume'].sum()],
    ['Small',len(small_wood_df),'<= 1','<= 0.1',small_wood_df['volume'].sum()]
]

print('')
print(tabulate(large_long_thick_small, headers=['Wood\nClass', 'Amount', 'Length\n(m)', 'Diameter\n(m)', 'Total Volume\n(m^3)']))
print('')








selected_columns_df_median = selected_columns_df_median.drop_duplicates(subset=['ID', 'vidnumber'], keep='first')
#   selected_columns_df_median = selected_columns_df_unique
print('LENGTH')
print(len(selected_columns_df_median))
print(selected_columns_df_median)
print('')

#median_values_df
smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] > 0.10]

large_wood_df = smaller_df[smaller_df['length_median'] > 1]

large_wood_df = large_wood_df.reset_index(drop=True)
large_wood_df['volume'] = large_wood_df['length_median'] * np.pi * ( large_wood_df['diameter_median'] / 2 ) ** 2

print('Large: '+str(len(large_wood_df)))



smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] <= 0.10]

semilarge_wood_df_long = smaller_df[smaller_df['length_median'] > 1]

semilarge_wood_df_long = semilarge_wood_df_long.reset_index(drop=True)
semilarge_wood_df_long['volume'] = semilarge_wood_df_long['length_median'] * np.pi * ( semilarge_wood_df_long['diameter_median'] / 2 ) ** 2

print('Long: '+str(len(semilarge_wood_df_long)))


smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] > 0.10]

semilarge_wood_df_thick = smaller_df[smaller_df['length_median'] <= 1]

semilarge_wood_df_thick = semilarge_wood_df_thick.reset_index(drop=True)
semilarge_wood_df_thick['volume'] = semilarge_wood_df_thick['length_median'] * np.pi * ( semilarge_wood_df_thick['diameter_median'] / 2 ) ** 2

print('Thick: '+str(len(semilarge_wood_df_thick)))


smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] <= 0.10]

small_wood_df = smaller_df[smaller_df['length_median'] <= 1]

small_wood_df = small_wood_df.reset_index(drop=True)
small_wood_df['volume'] = small_wood_df['length_median'] * np.pi * ( small_wood_df['diameter_median'] / 2 ) ** 2

print('Small: '+str(len(small_wood_df)))


large_long_thick_small = [
    ['Large',len(large_wood_df),'> 1','> 0.1',large_wood_df['volume'].sum()],
    ['Long',len(semilarge_wood_df_long),'> 1','<= 0.1',semilarge_wood_df_long['volume'].sum()],
    ['Thick',len(semilarge_wood_df_thick),'<= 1','> 0.1',semilarge_wood_df_thick['volume'].sum()],
    ['Small',len(small_wood_df),'<= 1','<= 0.1',small_wood_df['volume'].sum()]
]

print('')
print(tabulate(large_long_thick_small, headers=['Wood\nClass', 'Amount', 'Length\n(m)', 'Diameter\n(m)', 'Total Volume\n(m^3)']))
print('')
