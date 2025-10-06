import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
import os

# Define input and output folder
folder = 'allparts_data'

# Ensure output folder exists, create if not
if not os.path.exists(folder):
    os.makedirs(folder)

count = 1
while True:
    summary_file = folder+'/summary_ID'+str(count)+'.npy'
    count += 1
    if os.path.isfile(summary_file) == False:
        max_ID = count - 2
        print(max_ID)
        break

#max_ID = 51

# Lists to store akima trajectories
akima_trajectories = []

for id in range(1, max_ID + 1):
    print(id)
    name = f'summary_ID{id}'
    summary = np.load(os.path.join(folder, name + '.npy'))

    plt.figure(figsize=(10, 6))
    plt.plot(summary[:, 0], summary[:, 1], '.')

    # Generate a template array with all indices from the first to the last data point
    template_indices = np.arange(summary.shape[0])

    # Create an array with NaNs for the missing data points
    interpolated_summary = np.full((template_indices.shape[0], summary.shape[1]), np.nan)

    # Assign existing data to corresponding indices
    interpolated_summary[template_indices, :2] = summary[:, :2]

    # AKIMA interpolation
    x = np.arange(template_indices.min(), template_indices.max() + 1)
    
    # Mask out NaN values during interpolation
    mask = ~np.isnan(interpolated_summary[:, 0])
    akima_interp = Akima1DInterpolator(template_indices[mask], interpolated_summary[mask, :2], axis=0)
    
    vq_akima = akima_interp(x)
    akima_trajectories.append(vq_akima)
    
    # Save akima trajectory as .npy file
    np.save(os.path.join(folder, f'akima_trajectory_ID{id}.npy'), vq_akima)

    plt.plot(vq_akima[:, 0], vq_akima[:, 1], '--o', label='akima')
    plt.legend()
    plt.title(f'Interpolations for ID {id}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig(os.path.join(folder, f'interpolations_ID{id}.png'))
    plt.close()

# Plotting all trajectories for akima interpolation
plt.figure(figsize=(20, 12))
for trajectory in akima_trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-')
plt.title('All Trajectories (AKIMA Interpolation)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(os.path.join(folder, 'all_trajectories_akima.png'))
plt.close()
