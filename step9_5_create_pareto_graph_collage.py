import matplotlib.pyplot as plt
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

# Data for the top 20 pieces
df_top20 = df.head(20)

# Function to determine color based on conditions
def get_color(row):
    if row['length'] < 1 or row['diameter'] < 0.1:
        return 'red'
    else:
        return 'blue'

# Get colors for the top 500 pieces
colors_top500 = df_top500.apply(get_color, axis=1)

# Get colors for the top 20 pieces
colors_top20 = df_top20.apply(get_color, axis=1)

# Create figure with two side-by-side subplots
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Pareto for top 500 pieces with conditional coloring, filling the gaps
ax1.bar(df_top500['unique_ID'], df_top500['volume_median'], color=colors_top500, width=1)
ax2 = ax1.twinx()
ax2.plot(df_top500['unique_ID'], df_top500['Cumulative Percentage'], color='tab:orange')

# Customize Plot 1
ax1.set_xlabel('Piece ID')
ax1.set_ylabel('Volume (m³)')
ax2.set_ylabel('Cumulative Percentage of Volume')
ax1.set_title('Cumulative Wood Volumes (Top 500 Pieces)')

# Set X-axis with interval of 50, without labels
ax1.set_xticks(range(0, len(df_top500['unique_ID']), 50))
ax1.set_xticklabels(range(0, len(df_top500['unique_ID']), 50), rotation=45, ha='right')

# Plot 2: Pareto for top 20 pieces with conditional coloring
ax3.bar(df_top20['unique_ID'], df_top20['volume_median'], color=colors_top20)
ax4 = ax3.twinx()
ax4.plot(df_top20['unique_ID'], df_top20['Cumulative Percentage'], color='tab:orange', marker='D', ms=7)

# Customize Plot 2
ax3.set_xlabel('Piece ID')
ax3.set_ylabel('Volume (m³)')
ax4.set_ylabel('Cumulative Percentage of Volume')
ax4.set_ylim(0, 100)  # Start right Y-axis at 0 for the second graph
ax3.set_xticks(range(len(df_top20['unique_ID'])))
ax3.set_xticklabels(df_top20['unique_ID'], rotation=45, ha='right')
ax3.set_title('Cumulative Wood Volumes (Top 20 Pieces)')

# Adjust layout and save
plt.suptitle('Cumulative Wood Volumes')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(savefigfolder, 'combined_pareto_graph_side_by_side.png'))
plt.show()

