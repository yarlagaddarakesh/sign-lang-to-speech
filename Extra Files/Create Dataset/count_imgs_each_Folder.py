import os

# Path to the data_42 folder
data_42_folder = 'A:/Preprocess/Datasets/ASL'

# Initialize a dictionary to store the count of images in each folder
image_counts = {}

# Loop through the subdirectories in data_42_folder
for subdir in os.listdir(data_42_folder):
    # Construct the full path to the subdirectory
    subdir_path = os.path.join(data_42_folder, subdir)
    
    # Count the number of files (images) in the subdirectory
    num_images = len([name for name in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, name))])
    
    # Store the count of images in the dictionary
    image_counts[subdir] = num_images

# Display the count of images in each folder
for folder, count in image_counts.items():
    print(f"Folder '{folder}' contains {count} images.")

# Calculate total number of images across all folders
total_images = sum(image_counts.values())
print(f"Total number of images in all folders: {total_images}")
