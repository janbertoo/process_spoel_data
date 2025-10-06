import pickle
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # For Q-Q plot

vidnumbers = [1, 2, 3, 4, 5]

total_velocity_df = pd.DataFrame()

for vidnumber in vidnumbers:
    with open(f'step9_dataframe/dataframe_vid{vidnumber}_filtered_no_drone_overlap.p', 'rb') as file:
        data = pickle.load(file)
    
    df = pd.DataFrame(data)
    df['center_geo_x'] = pd.to_numeric(df['center_geo_x'], errors='coerce')
    df['center_geo_y'] = pd.to_numeric(df['center_geo_y'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.sort_values(by=['ID', 'timestamp'])
    
    flow_velocity_df = pd.read_csv(f'step6_LSPIV/all_vid{vidnumber}_05_speeds.csv', names=['X', 'Y', 'velocity'])
    flow_velocity_df['X'] = pd.to_numeric(flow_velocity_df['X'], errors='coerce')
    flow_velocity_df['Y'] = pd.to_numeric(flow_velocity_df['Y'], errors='coerce')
    flow_velocity_df['velocity'] = pd.to_numeric(flow_velocity_df['velocity'], errors='coerce')
    
    kdtree = KDTree(flow_velocity_df[['X', 'Y']].values)
    velocity_data = []

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if current_row['ID'] == next_row['ID'] and abs(current_row['timestamp'] - next_row['timestamp']) <= 0.3:
            velocity_log = np.sqrt((next_row['center_geo_x'] - current_row['center_geo_x'])**2 + 
                                   (next_row['center_geo_y'] - current_row['center_geo_y'])**2) / 0.25
            mid_x = (current_row['center_geo_x'] + next_row['center_geo_x']) / 2
            mid_y = (current_row['center_geo_y'] + next_row['center_geo_y']) / 2
            _, idx = kdtree.query([mid_x, mid_y])
            closest_flow_velocity = float(flow_velocity_df.iloc[idx]['velocity'])

            velocity_data.append({
                'vid_number': vidnumber,
                'ID': current_row['ID'],
                'length_median': current_row['length_median'],
                'diameter_median': current_row['diameter_median'],
                'velocity_log': velocity_log,
                'closest_flow_velocity': closest_flow_velocity
            })
    
    velocity_df = pd.DataFrame(velocity_data)
    total_velocity_df = pd.concat([total_velocity_df, velocity_df])

total_velocity_df = total_velocity_df[total_velocity_df['length_median'] >= 1]
total_velocity_df = total_velocity_df[total_velocity_df['diameter_median'] >= 0.1]
total_velocity_df = total_velocity_df.dropna(subset=['velocity_log', 'diameter_median'])

# Normality Tests and Q-Q Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(total_velocity_df['velocity_log'], bins=30, kde=True)
plt.xlabel('Log Velocity (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of Log Velocity')

plt.subplot(1, 2, 2)
sm.qqplot(total_velocity_df['velocity_log'], line='s')
plt.title('Q-Q Plot of Log Velocity')
plt.tight_layout()
plt.savefig('step10_analyze_rotation/qqplot_velocity_log.pdf')
plt.show()

# Shapiro-Wilk Test for Normality
shapiro_test = stats.shapiro(total_velocity_df['velocity_log'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.3f}, P-value: {shapiro_test.pvalue:.3f}")

# Pearson correlation
pearson_corr, pearson_p_value = stats.pearsonr(total_velocity_df['diameter_median'], total_velocity_df['velocity_log'])
print(f"Pearson correlation coefficient: {pearson_corr:.3f}, P-value: {pearson_p_value:.3f}")

# Spearman correlation
spearman_corr, spearman_p_value = stats.spearmanr(total_velocity_df['diameter_median'], total_velocity_df['velocity_log'])
print(f"Spearman correlation coefficient: {spearman_corr:.3f}, P-value: {spearman_p_value:.3f}")
