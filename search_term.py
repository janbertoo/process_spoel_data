import os
import re

# Define the folder path and the term to search for
folder_path = '/home/jean-pierre/ownCloud/phd/code_code_code_code_code/'  # Replace with your folder path
folder_path = os.getcwd()
search_term = 'kde_plot_velocity_vs_rotation_no_averageline'  # Replace with the term you're searching for

# Loop through all files in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Only search Python files
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Search for the term in the file
                if re.search(search_term, content):
                    print(f"Found '{search_term}' in {file_path}")

