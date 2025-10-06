import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ast
from tabulate import tabulate

savefigfolder = 'step9_dataframe/'

vidnumbers = [1,2,3,4,5,'ALL']

def create_images_pieces_and_bin(df, savefigfolder, vidnumber, dronenumbers):
	################################################################################# PLOT LENGTHS AND DIAMETERS OF INDIVIDUAL PIECES
	# Create subplots
	fig, axes = plt.subplots(2, 1, figsize=(20, 10))

	df_unique = df.drop_duplicates(subset=['ID', 'vidnumber'], keep='first')
	df_unique = df_unique.reset_index(drop=True)

	# Plot length_median
	axes[0].bar(df_unique.index, df_unique['length_median'], color='blue')
	axes[0].set_title('Median Length')
	axes[0].set_ylabel('Length (m)')
	axes[0].axhline(y=1, color='red', linestyle='--')  # Add horizontal line at 1m

	# Plot diameter_median
	axes[1].bar(df_unique.index, df_unique['diameter_median'], color='orange')
	axes[1].set_title('Median Diameter')
	axes[1].set_ylabel('Diameter (m)')
	axes[1].axhline(y=0.1, color='red', linestyle='--')  # Add horizontal line at 0.1m

	# Adjust layout
	plt.tight_layout()

	# Show plot
	plt.savefig(savefigfolder+'lengths_and_diameters_vid'+str(vidnumber)+'_drone'+str(dronenumbers)+'.png')
	plt.savefig(savefigfolder+'lengths_and_diameters_vid'+str(vidnumber)+'_drone'+str(dronenumbers)+'.eps')

	################################################################################# PLOT LENGTHS AND DIAMETERS IN BINS

	# Define bin edges
	#print(df_unique)
	length_bins = np.arange(0, df_unique['length_median'].max() + 0.25, 0.25)
	diameter_bins = np.arange(0, df_unique['diameter_median'].max() + 0.025, 0.025)

	# Create subplots
	fig, axes = plt.subplots(1, 2, figsize=(15, 7))

	# Plot length_median histogram
	axes[0].hist(df_unique['length_median'], bins=length_bins, color='blue')
	axes[0].set_title('Median Length Histogram')
	axes[0].set_ylabel('Frequency')
	axes[0].set_xlabel('Length (m)')
	axes[0].axvline(x=1, color='red', linestyle='--')  # Add vertical line at 1m

	# Plot diameter_median histogram
	axes[1].hist(df_unique['diameter_median'], bins=diameter_bins, color='orange')
	axes[1].set_title('Median Diameter Histogram')
	axes[1].set_ylabel('Frequency')
	axes[1].set_xlabel('Diameter (m)')
	axes[1].axvline(x=0.1, color='red', linestyle='--')  # Add vertical line at 0.1m

	# Adjust layout
	plt.tight_layout()

	# Show plot
	plt.savefig(savefigfolder+'lengths_and_diameters_bins_vid'+str(vidnumber)+'_drone'+str(dronenumbers)+'.png')
	plt.savefig(savefigfolder+'lengths_and_diameters_bins_vid'+str(vidnumber)+'_drone'+str(dronenumbers)+'.eps')

	return df_unique

def create_table_large_long_thick_small(df_unique):
	selected_columns_df_median = df_unique

	smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] > 0.10]

	large_wood_df = smaller_df[smaller_df['length_median'] > 1]

	large_wood_df = large_wood_df.reset_index(drop=True)
	large_wood_df['volume'] = large_wood_df['length_median'] * np.pi * ( large_wood_df['diameter_median'] / 2 ) ** 2

	#print('Large: '+str(len(large_wood_df)))



	smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] <= 0.10]

	semilarge_wood_df_long = smaller_df[smaller_df['length_median'] > 1]

	semilarge_wood_df_long = semilarge_wood_df_long.reset_index(drop=True)
	semilarge_wood_df_long['volume'] = semilarge_wood_df_long['length_median'] * np.pi * ( semilarge_wood_df_long['diameter_median'] / 2 ) ** 2

	#print('Long: '+str(len(semilarge_wood_df_long)))


	smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] > 0.10]

	semilarge_wood_df_thick = smaller_df[smaller_df['length_median'] <= 1]

	semilarge_wood_df_thick = semilarge_wood_df_thick.reset_index(drop=True)
	semilarge_wood_df_thick['volume'] = semilarge_wood_df_thick['length_median'] * np.pi * ( semilarge_wood_df_thick['diameter_median'] / 2 ) ** 2

	#print('Thick: '+str(len(semilarge_wood_df_thick)))


	smaller_df = selected_columns_df_median[selected_columns_df_median['diameter_median'] <= 0.10]

	small_wood_df = smaller_df[smaller_df['length_median'] <= 1]

	small_wood_df = small_wood_df.reset_index(drop=True)
	small_wood_df['volume'] = small_wood_df['length_median'] * np.pi * ( small_wood_df['diameter_median'] / 2 ) ** 2

	#print('Small: '+str(len(small_wood_df)))


	large_long_thick_small = [
	    ['Large',len(large_wood_df),'> 1','> 0.1',large_wood_df['volume'].sum()],
	    ['Long',len(semilarge_wood_df_long),'> 1','<= 0.1',semilarge_wood_df_long['volume'].sum()],
	    ['Thick',len(semilarge_wood_df_thick),'<= 1','> 0.1',semilarge_wood_df_thick['volume'].sum()],
	    ['Small',len(small_wood_df),'<= 1','<= 0.1',small_wood_df['volume'].sum()]
	]

	print('')
	print(tabulate(large_long_thick_small, headers=['Wood\nClass', 'Amount', 'Length\n(m)', 'Diameter\n(m)', 'Total Volume\n(m^3)']))
	print('')



for vidnumber in vidnumbers:
	print('')
	print(vidnumber)
	
	file_path = os.path.join(savefigfolder,'dataframe_vid' + str(vidnumber) + '.p')
	df = pd.read_pickle(file_path)
	#print(df)

	print('')
	print('Drone ALL:')
	df_unique = create_images_pieces_and_bin(df, savefigfolder, vidnumber, 'ALL')
	create_table_large_long_thick_small(df_unique)

	
	print('Drone 1:')
	df_drone1 = df[df['drone_number'] == 1]
	df_unique = create_images_pieces_and_bin(df_drone1, savefigfolder, vidnumber,'1')
	create_table_large_long_thick_small(df_unique)

	print('Drone 6:')
	df_drone6 = df[df['drone_number'] == 6]
	df_unique = create_images_pieces_and_bin(df_drone6, savefigfolder, vidnumber,'6')
	create_table_large_long_thick_small(df_unique)

	print('Drone 16:')
	df_drone16 = df[df['drone_number'] == 16]
	df_unique = create_images_pieces_and_bin(df_drone16, savefigfolder, vidnumber,'16')
	create_table_large_long_thick_small(df_unique)
	
	

	print('')
	print('')
	print('')
	print('')
	print('')