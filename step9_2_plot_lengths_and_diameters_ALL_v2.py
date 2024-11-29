import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

savefigfolder = 'step9_dataframe/'

vidnumbers = [1, 2, 3, 4, 5]

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

df = combine_dataframes(savefigfolder, vidnumbers)

# Create a new dataframe with selected columns
selected_columns_df = df[['ID', 'vidnumber', 'length', 'diameter']]

# Group by 'ID' and calculate the median for each group
median_values_df = selected_columns_df.groupby(['ID']).median()
median_values_df['volume'] = median_values_df['length'] * np.pi * (median_values_df['diameter'] / 2) ** 2

# Reindex
median_values_df = median_values_df.reset_index(drop=True)

#################################################################################
# CALCULATE AVERAGE VALUES
average_length = median_values_df['length'].mean()
average_diameter = median_values_df['diameter'].mean()
average_volume = median_values_df['volume'].mean()

#################################################################################
# PLOT LENGTHS, DIAMETERS, AND VOLUMES IN BINS

# Define bin edges
length_bins = np.arange(0, median_values_df['length'].max() + 0.25, 0.25)  # Original bin size
diameter_bins = np.arange(0, median_values_df['diameter'].max() + 0.025, 0.025)  # Original bin size
volume_bins = np.arange(0, median_values_df['volume'].max() + 0.01, 0.01)  # Double bin size for volumes

# Create a figure for the length histogram
fig3, ax3 = plt.subplots(figsize=(4, 4))

# Plot length histogram
ax3.hist(median_values_df['length'], bins=length_bins, color='tab:blue', label='Length Distribution')
#ax3.set_title('Length Histogram')
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Length (m)')

# Add vertical lines: large wood requirement (1m) and average length
ax3.axvline(x=1, color='red', linestyle='--', label='Large Wood (1m)')
ax3.axvline(x=average_length, color='blue', linestyle='-', label=f'Average ({average_length:.3f}m)')

# Add legend
ax3.legend()

# Save the length histogram with the average line
plt.tight_layout()
plt.savefig(savefigfolder + 'lengths_histogram_vidALL.png')
plt.savefig(savefigfolder + 'lengths_histogram_vidALL.pdf')

# Create a figure for the diameter histogram
fig4, ax4 = plt.subplots(figsize=(4, 4))

# Plot diameter histogram
ax4.hist(median_values_df['diameter'], bins=diameter_bins, color='tab:orange', label='Diameter Distribution')
#ax4.set_title('Diameter Histogram')
ax4.set_ylabel('Frequency')
ax4.set_xlabel('Diameter (m)')

ax4.set_xlim(0,0.4)

# Add vertical lines: large wood requirement (0.1m) and average diameter
ax4.axvline(x=0.1, color='red', linestyle='--', label='Large Wood (0.1m)')
ax4.axvline(x=average_diameter, color='blue', linestyle='-', label=f'Average ({average_diameter:.3f}m)')

# Add legend
ax4.legend()

# Save the diameter histogram with the average line
plt.tight_layout()
plt.savefig(savefigfolder + 'diameters_histogram_vidALL.png')
plt.savefig(savefigfolder + 'diameters_histogram_vidALL.pdf')

# Create a figure for the volume histogram
fig5, ax5 = plt.subplots(figsize=(4, 4))

# Plot volume histogram
ax5.hist(median_values_df['volume'], bins=volume_bins, color='tab:green', label='Volume Distribution')
#ax5.set_title('Volume Histogram')
ax5.set_ylabel('Frequency')
ax5.set_xlabel('Volume (m³)')

# Add a red dotted line at 1 * pi * 0.05^2
red_line_value = 1 * np.pi * (0.05 ** 2)
ax5.axvline(x=red_line_value, color='red', linestyle='--', label=f'Large Wood ({red_line_value:.6f} m³)')
# Add vertical lines for the average volume
ax5.axvline(x=average_volume, color='blue', linestyle='-', label=f'Average ({average_volume:.6f} m³)')



# Add legend
ax5.legend()

# Save the volume histogram with the average and reference lines
plt.tight_layout()
plt.savefig(savefigfolder + 'volumes_histogram_vidALL.png')
plt.savefig(savefigfolder + 'volumes_histogram_vidALL.pdf')

