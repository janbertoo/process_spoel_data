import pandas as pd
import matplotlib.pyplot as plt
import os

savefigfolder = 'step9_dataframe/'
file_path = os.path.join(savefigfolder,'dataframe_vidALL.p')
df = pd.read_pickle(file_path)
print(df)
df['unique_ID'] = df['vidnumber'].astype(str) + '.' + df['ID'].astype(str)
df = df.drop_duplicates(subset=['ID', 'vidnumber'], keep='first')
print(df)
df = df.sort_values(by='volume_median', ascending=False)
df['Cumulative Volume'] = df['volume_median'].cumsum()
df['Cumulative Percentage'] = 100 * df['Cumulative Volume'] / df['volume_median'].sum()

# Limit the DataFrame to the top 20 rows
df_top20 = df#.head(20)

# Plotting
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(df_top20['unique_ID'], df_top20['volume_median'], color='tab:blue')
ax2 = ax.twinx()
ax2.plot(df_top20['unique_ID'], df_top20['Cumulative Percentage'], color='tab:orange', marker='D', ms=7)

# Setting x-axis limits
#ax.set_xlim(-0.5, 19.5)  # Adjusting to display only the top 20
#ax2.set_ylim(0, 40)  # Assuming the cumulative percentage goes from 0 to 100

# Update y-axis label with m³ superscript
ax.set_xlabel('Piece ID')
ax.set_ylabel('Volume (m³)')  # Use superscript ³
ax2.set_ylabel('Cumulative Percentage of Volume')

# Set x-tick labels with rotation
ax.set_xticks(range(len(df_top20['unique_ID'])))
ax.set_xticklabels(df_top20['unique_ID'], rotation=45, ha='right')

plt.title('Cumulative Wood Volumes')
fig.tight_layout()

# Saving the figures
plt.savefig(os.path.join(savefigfolder,'pareto_graph_piece_volumes_all.eps'))
plt.savefig(os.path.join(savefigfolder,'pareto_graph_piece_volumes_all.png'))
plt.savefig(os.path.join(savefigfolder,'pareto_graph_piece_volumes_all.pdf'))

df_2_38 = df[df['unique_ID'] == '2.38']
df_5_28 = df[df['unique_ID'] == '5.28']
print(df_2_38)
print(df_5_28)
