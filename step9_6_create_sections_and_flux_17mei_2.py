import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

savefigfolder = 'step9_dataframe/'
sectionsfolder = '/home/jean-pierre/ownCloud/phd/spoel_data_2023/GPS_Data-20230619T123957Z-001/GPS_Data/'
file_path = os.path.join(savefigfolder,'dataframe_vidALL_filtered_no_drone_overlap_shifted.p')
df = pd.read_pickle(file_path)
print(df)

size_of_bins = 1

# Define base coordinates
Xbase = 2803987
Ybase = 1175193

base_ruler = 1489.126

#Section1 = ((2804162.1299,1175290.8655),(2804181.03779,1175253.16418))
#Section2 = ((2804132.9050,1175280.4683),(2804142.0906,1175249.4849))
#Section3 = ((2804104.0364,1175274.5377),(2804106.5156,1175248.0643))
#Section4 = ((2804056.8109,1175273.5760),(2804070.4716,1175236.7660))
#Section5 = ((2804033.1186,1175264.8710),(2804059.1070,1175228.7094))
#Section6 = ((2804005.0938,1175250.3824),(2804041.5581,1175221.2789))

Section6 = ((2804162.1299-Xbase,1175290.8655-Ybase),(2804181.03779-Xbase,1175253.16418-Ybase))
Section5 = ((2804132.9050-Xbase,1175280.4683-Ybase),(2804142.0906-Xbase,1175249.4849-Ybase))
Section4 = ((2804104.0364-Xbase,1175274.5377-Ybase),(2804106.5156-Xbase,1175248.0643-Ybase))
Section3 = ((2804056.8109-Xbase,1175273.5760-Ybase),(2804070.4716-Xbase,1175236.7660-Ybase))
Section2 = ((2804033.1186-Xbase,1175264.8710-Ybase),(2804059.1070-Xbase,1175228.7094-Ybase))
Section1 = ((2804005.0938-Xbase,1175250.3824-Ybase),(2804041.5581-Xbase,1175221.2789-Ybase))

#water levels video 1
#water_levels = [1487.88, 1488.2, 1488.662, 1488.95, 1489.55, 1490.03]
water_levels_base = [1488.20, 1488.52, 1488.982, 1489.27, 1489.87, 1490.35]
water_levels_base = [1488.59, 1488.91, 1489.37, 1489.66, 1490.26, 1490.74]

colors = ['red', 'purple', 'orange', 'cyan', 'b']
vid_lengths_section_1_2 = [3218 / 4, ]
vid_lengths_section_3_4 = [3218 / 4, ]
vid_lengths_section_5_6 = [3218 / 4, ]

#water level corrections
corrections_and_vidnumber = [[0.39,1],[0.42,2],[0.42,3],[0.45,4],[0.47,5]]
corrections_and_vidnumber = [[0,1,colors[0]],[0.03,2,colors[1]],[0.03,3,colors[2]],[0.06,4,colors[3]],[0.08,5,colors[4]]]


pre_data_files = [sectionsfolder+'presection1.csv',sectionsfolder+'presection2.csv',sectionsfolder+'presection3.csv',sectionsfolder+'presection4.csv',sectionsfolder+'presection5.csv',sectionsfolder+'presection6.csv']
post_data_files = [sectionsfolder+'postsection1.csv',sectionsfolder+'postsection2.csv',sectionsfolder+'postsection3.csv',sectionsfolder+'postsection4.csv',sectionsfolder+'postsection5.csv',sectionsfolder+'postsection6.csv']

sections_and_water_levels = [
    [Section1,1,water_levels_base[0],pre_data_files[0],post_data_files[0]],
    [Section2,2,water_levels_base[1],pre_data_files[1],post_data_files[1]],
    [Section3,3,water_levels_base[2],pre_data_files[2],post_data_files[2]],
    [Section4,4,water_levels_base[3],pre_data_files[3],post_data_files[3]],
    [Section5,5,water_levels_base[4],pre_data_files[4],post_data_files[4]],
    [Section6,6,water_levels_base[5],pre_data_files[5],post_data_files[5]],  
    ]

def line_intersection(line1, line2):
    #print('line1 and line2')
    #print(line1)
    #print(line2)
    if line1 == None or line2 == None:
        return None

    def line_coefficients(p1, p2):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return A, B, C

    def is_between(a, b, c):
        """Return True if point c is between points a and b (inclusive)."""
        return min(a, b) <= c <= max(a, b)

    A1, B1, C1 = line_coefficients(line1[0], line1[1])
    A2, B2, C2 = line_coefficients(line2[0], line2[1])
    
    determinant = A1 * B2 - A2 * B1
    
    if determinant == 0:
        return None  # Lines are parallel and don't intersect
    
    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    
    # Check if the intersection point (x, y) is within the bounding box of line1
    if (is_between(line1[0][0], line1[1][0], x) and 
        is_between(line1[0][1], line1[1][1], y)):
        #print('FOUND INTERSECTION')
        return x, y
    
    return None

def project_point_on_line(point, line):
    def line_coefficients(p1, p2):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return A, B, C

    def intersection_point(A1, B1, C1, A2, B2, C2):
        determinant = A1 * B2 - A2 * B1
        if determinant == 0:
            return None  # Lines are parallel and don't intersect
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return x, y

    A1, B1, C1 = line_coefficients(line[0], line[1])
    
    # For the perpendicular line passing through the point (x0, y0)
    x0, y0 = point
    A2 = -B1
    B2 = A1
    C2 = A2 * x0 + B2 * y0
    
    return intersection_point(A1, B1, C1, A2, B2, C2)

def calc_distance_to_begin_section(point, line):
	if point == None:
		return None
	
	else:
		begin_line = line[0]

		distance = np.sqrt( ( point[0] - begin_line[0] ) **2 + ( point[1] - begin_line[1] ) ** 2 )

		return distance

# Function to create the line tuple
def create_line(row):
    if isinstance(row['center_geo'], list) and isinstance(row['center_before'], list):
        return (tuple(row['center_geo']), tuple(row['center_before']))
    else:
        return None



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming functions project_point_on_line and calc_distance_to_begin_section are defined elsewhere
# Assuming Xbase and Ybase are defined elsewhere
# Assuming sections_and_water_levels and corrections_and_vidnumber are defined elsewhere
# Assuming create_line and line_intersection are defined elsewhere
# Assuming df is defined elsewhere and contains the necessary data

for section, section_number, water_level_base, pre_section_file, post_section_file in sections_and_water_levels:
    print('')
    print('Section: ' + str(section))
    
    df_pre_section = pd.read_csv(pre_section_file)
    df_pre_section['X'] = df_pre_section['X'] - Xbase
    df_pre_section['Y'] = df_pre_section['Y'] - Ybase
    df_pre_section['project'] = df_pre_section.apply(lambda row: project_point_on_line((row['X'], row['Y']), section), axis=1)
    df_pre_section['distance_from_base'] = df_pre_section.apply(lambda row: calc_distance_to_begin_section((row['project']), section), axis=1)
    df_pre_section['Z'] = df_pre_section['Z'] - water_level_base
    
    if section_number == 1:
        df_pre_section['Z'] = df_pre_section['Z'] + 0.245 - ((0.245 + 0.273) / 45) * df_pre_section['distance_from_base']
    
    df_pre_section = df_pre_section.sort_values('distance_from_base')
    
    df_post_section = pd.read_csv(post_section_file)
    df_post_section['X'] = df_post_section['X'] - Xbase
    df_post_section['Y'] = df_post_section['Y'] - Ybase
    df_post_section['project'] = df_post_section.apply(lambda row: project_point_on_line((row['X'], row['Y']), section), axis=1)
    df_post_section['distance_from_base'] = df_post_section.apply(lambda row: calc_distance_to_begin_section((row['project']), section), axis=1)
    df_post_section['Z'] = df_post_section['Z'] - water_level_base
    
    # Create a figure with one main subplot and additional subplots for bar charts
    num_subplots = len(corrections_and_vidnumber) + 1  # +1 for the main plot
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots), sharex=True)
    
    # Main plot (first subplot)
    ax = axs[0]
    ax.fill_between(df_pre_section['distance_from_base'], df_pre_section['Z'], where=(df_pre_section['Z'] < 0), interpolate=True, color='blue', alpha=0.1)
    ax.fill_between(df_post_section['distance_from_base'], df_post_section['Z'], where=(df_post_section['Z'] < 0), interpolate=True, color='blue', alpha=0.1)
    ax.plot(df_pre_section['distance_from_base'], df_pre_section['Z'], color='green', label='Pre Flood Section')
    ax.plot(df_post_section['distance_from_base'], df_post_section['Z'], color='brown', label='Post Flood Section')
    
    ax.set_ylabel('Z (m)')
    ax.legend()
    ax.set_title(f'Section {section_number}')
    
    # Additional subplots for bar charts
    max_y_value = 0  # Initialize the maximum y-value
    
    for i, (correction, vidnumber, color) in enumerate(corrections_and_vidnumber):
        df_vid = df[df['vidnumber'] == vidnumber]
        df_vid['line'] = df_vid.apply(create_line, axis=1)
        df_vid['line_intersect'] = df_vid['line'].apply(lambda x: line_intersection(x, section))
        df_vid['line_intersect_distance_from_base'] = df_vid['line_intersect'].apply(lambda x: calc_distance_to_begin_section(x, section))
        df_vid = df_vid[df_vid['line_intersect_distance_from_base'].notna()]
        
        weighted_centers = df_vid['line_intersect_distance_from_base'] * df_vid['volume_median']
        total_volume = df_vid['volume_median'].sum()
        weighted_average = weighted_centers.sum() / total_volume
        
        ax2 = axs[i + 1]
        
        bin_edges = np.arange(0, int(df_vid['line_intersect_distance_from_base'].max()) + 2, size_of_bins)
        binned_values, _ = np.histogram(df_vid['line_intersect_distance_from_base'], bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        volume_sum = []
        for j in range(len(bin_edges) - 1):
            mask = (df_vid['line_intersect_distance_from_base'] >= bin_edges[j]) & (df_vid['line_intersect_distance_from_base'] < bin_edges[j + 1])
            volume = df_vid.loc[mask, 'volume_median'].sum()
            volume_sum.append(volume)
        
        ax2.bar(bin_centers, volume_sum, align='center', width=size_of_bins, color=color)
        ax2.axvline(x=weighted_average, color='black', linestyle='--', label='Weighted Average',linewidth=2)
        
        ax2.set_ylabel('Volume (m^3)')
        ax2.legend()
        ax2.set_title(f'Video {vidnumber}, Total Volume {round(total_volume, 2)}')
        
        # Update max_y_value if necessary
        max_volume = max(volume_sum)
        if max_volume > max_y_value:
            max_y_value = max_volume
    
    # Set the y-axis limit for all additional subplots
    for ax2 in axs[1:]:
        ax2.set_ylim(0, max_y_value)
    
    # Set the x-axis label for the last subplot
    axs[-1].set_xlabel('Distance from North Bank (m)')
    
    # Draw axvlines on the main plot
    for correction, vidnumber, color in corrections_and_vidnumber:
        df_vid = df[df['vidnumber'] == vidnumber]
        df_vid['line'] = df_vid.apply(create_line, axis=1)
        df_vid['line_intersect'] = df_vid['line'].apply(lambda x: line_intersection(x, section))
        df_vid['line_intersect_distance_from_base'] = df_vid['line_intersect'].apply(lambda x: calc_distance_to_begin_section(x, section))
        df_vid = df_vid[df_vid['line_intersect_distance_from_base'].notna()]
        
        weighted_centers = df_vid['line_intersect_distance_from_base'] * df_vid['volume_median']
        total_volume = df_vid['volume_median'].sum()
        weighted_average = weighted_centers.sum() / total_volume
        
        ax.axvline(x=weighted_average, color=color, linestyle='--', label=f'Correction {correction}',linewidth=2)
    
    #plt.tight_layout()
    plt.show()



















'''



for section, water_level_base in sections_and_water_levels:
    for correction in corrections:
    	print(section)
    	water_level = water_levels_base + correction
    	print(water_level)

    	df_intersect = df



# Example usage:
line = ((1, 2), (3, 1))
line2 = ((1, 1), (4, 2))
intersection = line_intersection(line, line2)
distance = calc_distance_to_begin_section(intersection, line)
print(distance)

# Example usage:
#line = ((1, 2), (3, 1))
point = (1, 1)
projection = project_point_on_line(point, line)
distance = calc_distance_to_begin_section(projection, line)

print(distance)

# Example usage:
line2 = ((1, 1), (2, 0.75))
intersection = line_intersection(line, line2)
print(intersection)
distance = calc_distance_to_begin_section(intersection, line)
print(distance)

'''