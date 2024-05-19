import os

# Path to the directory containing folders
directory = 'A:/Preprocess/collect images/Without Landmarks'

# Get list of all folders in the directory
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Iterate through each folder and rename
for folder in folders:
    # Convert the folder name to integer and add 36
    new_name = str(int(folder) + 36)
    
    # Construct the full path of the old and new folder names
    old_path = os.path.join(directory, folder)
    new_path = os.path.join(directory, new_name)
    
    # Rename the folder
    os.rename(old_path, new_path)
    
    print(f"Renamed {folder} to {new_name}")

print("All folders renamed successfully.")
