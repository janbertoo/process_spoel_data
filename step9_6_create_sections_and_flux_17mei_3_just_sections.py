import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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



# Define necessary parameters
savefigfolder = 'step9_dataframe/'
sectionsfolder = '/home/jean-pierre/ownCloud/phd/spoel_data_2023/GPS_Data-20230619/GPS_Data/'
file_path = os.path.join(savefigfolder, 'dataframe_vidALL_filtered_no_drone_overlap_shifted.p')
df = pd.read_pickle(file_path)

# Define the sections and other parameters
sections_and_water_levels = [
    [((2804162.1299 - 2803987, 1175290.8655 - 1175193), (2804181.03779 - 2803987, 1175253.16418 - 1175193)), 1, 1490.74, sectionsfolder + 'presection1.csv', sectionsfolder + 'postsection1.csv'],
    [((2804132.9050 - 2803987, 1175280.4683 - 1175193), (2804142.0906 - 2803987, 1175249.4849 - 1175193)), 2, 1490.26, sectionsfolder + 'presection2.csv', sectionsfolder + 'postsection2.csv'],
    [((2804104.0364 - 2803987, 1175274.5377 - 1175193), (2804106.5156 - 2803987, 1175248.0643 - 1175193)), 3, 1489.66, sectionsfolder + 'presection3.csv', sectionsfolder + 'postsection3.csv'],
    [((2804056.8109 - 2803987, 1175273.5760 - 1175193), (2804070.4716 - 2803987, 1175236.7660 - 1175193)), 4, 1489.37, sectionsfolder + 'presection4.csv', sectionsfolder + 'postsection4.csv'],
    [((2804033.1186 - 2803987, 1175264.8710 - 1175193), (2804059.1070 - 2803987, 1175228.7094 - 1175193)), 5, 1488.91, sectionsfolder + 'presection5.csv', sectionsfolder + 'postsection5.csv'],
    [((2804005.0938 - 2803987, 1175250.3824 - 1175193), (2804041.5581 - 2803987, 1175221.2789 - 1175193)), 6, 1488.59, sectionsfolder + 'presection6.csv', sectionsfolder + 'postsection6.csv']
]

colors = ['red', 'purple', 'orange', 'cyan', 'b']
corrections_and_vidnumber = [[0, 1, colors[0]], [0.03, 2, colors[1]], [0.03, 3, colors[2]], [0.06, 4, colors[3]], [0.08, 5, colors[4]]]

fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

# Loop through each section
for i, (section, section_number, water_level_base, pre_section_file, post_section_file) in enumerate(sections_and_water_levels):
    ax = axs[i]

    # Load pre-flood data
    df_pre_section = pd.read_csv(pre_section_file)
    df_pre_section['X'] -= 2803987
    df_pre_section['Y'] -= 1175193
    df_pre_section['distance_from_base'] = df_pre_section.apply(lambda row: calc_distance_to_begin_section(project_point_on_line((row['X'], row['Y']), section), section), axis=1)
    df_pre_section['Z'] -= water_level_base

    # Load post-flood data
    df_post_section = pd.read_csv(post_section_file)
    df_post_section['X'] -= 2803987
    df_post_section['Y'] -= 1175193
    df_post_section['distance_from_base'] = df_post_section.apply(lambda row: calc_distance_to_begin_section(project_point_on_line((row['X'], row['Y']), section), section), axis=1)
    df_post_section['Z'] -= water_level_base

    # Plot pre- and post-flood cross-sections
    ax.plot(df_pre_section['distance_from_base'], df_pre_section['Z'], color='green', label='Pre Flood')
    ax.plot(df_post_section['distance_from_base'], df_post_section['Z'], color='brown', label='Post Flood')
    ax.fill_between(df_pre_section['distance_from_base'], df_pre_section['Z'], where=(df_pre_section['Z'] < 0), interpolate=True, color='blue', alpha=0.1)
    ax.fill_between(df_post_section['distance_from_base'], df_post_section['Z'], where=(df_post_section['Z'] < 0), interpolate=True, color='blue', alpha=0.1)

    # Add average wood transport location lines
    for correction, vidnumber, color in corrections_and_vidnumber:
        df_vid = df[df['vidnumber'] == vidnumber]
        df_vid['line'] = df_vid.apply(create_line, axis=1)
        df_vid['line_intersect'] = df_vid['line'].apply(lambda x: line_intersection(x, section))
        df_vid['line_intersect_distance_from_base'] = df_vid['line_intersect'].apply(lambda x: calc_distance_to_begin_section(x, section))
        df_vid = df_vid[df_vid['line_intersect_distance_from_base'].notna()]

        weighted_centers = df_vid['line_intersect_distance_from_base'] * df_vid['volume_median']
        total_volume = df_vid['volume_median'].sum()
        if total_volume > 0:
            weighted_average = weighted_centers.sum() / total_volume
            ax.axvline(x=weighted_average, color=color, linestyle='--', linewidth=2)

    ax.set_ylabel('Z (m)')
    ax.set_title(f'Section {section_number}')
    ax.legend()

axs[-1].set_xlabel('Distance from North Bank (m)')

plt.tight_layout()
plt.savefig(savefigfolder + 'combined_cross_sections.png')
plt.savefig(savefigfolder + 'combined_cross_sections.pdf')
plt.show()
