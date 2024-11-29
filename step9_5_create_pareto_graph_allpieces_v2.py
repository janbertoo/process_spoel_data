import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

# Load data (modify the path as necessary)
savefigfolder = 'step9_dataframe/'
file_path = os.path.join(savefigfolder, 'dataframe_vidALL.p')
df = pd.read_pickle(file_path)

# Convert 'length' and 'diameter' columns to numeric, forcing errors to NaN
df['length'] = pd.to_numeric(df['length'], errors='coerce')
df['diameter'] = pd.to_numeric(df['diameter'], errors='coerce')

# Prepare Data
df['unique_ID'] = df['vidnumber'].astype(str) + '.' + df['ID'].astype(str)
df = df.drop_duplicates(subset=['ID', 'vidnumber'], keep='first')
df = df.sort_values(by='volume_median', ascending=False)
df['Cumulative Volume'] = df['volume_median'].cumsum()
df['Cumulative Percentage'] = 100 * df['Cumulative Volume'] / df['volume_median'].sum()

# Data for the top 500 pieces
df_top500 = df.head(500)

# Function to determine color based on conditions
def get_color(row):
    if row['length'] < 1 or row['diameter'] < 0.1:
        return 'red'
    else:
        return 'tab:blue'

# Get colors for the top 500 pieces
colors_top500 = df_top500.apply(get_color, axis=1)

# Create figure for the top graph
fig, ax1 = plt.subplots(figsize=(8, 4))

# Plot: Pareto for top 500 pieces with conditional coloring, filling the gaps
ax1.bar(df_top500['unique_ID'], df_top500['volume_median'], color=colors_top500, width=1)
ax2 = ax1.twinx()
ax2.plot(df_top500['unique_ID'], df_top500['Cumulative Percentage'], color='tab:orange')

# Customize Plot
ax1.set_xlabel('Log')
ax1.set_ylabel('Volume (m³)')
ax2.set_ylabel('Cumulative Percentage of Volume')
#ax1.set_title('Cumulative Wood Volumes')

# Set X-axis with interval of 50, without labels
ax1.set_xticks(range(0, len(df_top500['unique_ID']), 50))
ax1.set_xticklabels(range(0, len(df_top500['unique_ID']), 50), rotation=45, ha='right')

# Add legend for color coding, positioned slightly inside on the right
red_patch = mpatches.Patch(color='red', label='Volume (m³), Length < 1m or Diameter < 0.1m')
blue_patch = mpatches.Patch(color='tab:blue', label='Volume (m³), Length ≥ 1m and Diameter ≥ 0.1m')
orange_line = mpatches.Patch(color='tab:orange', label='Cumulative Percentage')

plt.legend(handles=[red_patch, blue_patch, orange_line], loc='center left', bbox_to_anchor=(0.35, 0.4), title="Legend")

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(savefigfolder, 'pareto_graph_top_500_centered_legend_inside.png'))
plt.savefig(os.path.join(savefigfolder, 'pareto_graph_top_500_centered_legend_inside.pdf'))
plt.show()
