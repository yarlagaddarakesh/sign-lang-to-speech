import os

def compare_and_remove(folder1, folder2):
    subfolders1 = os.listdir(folder1)
    subfolders2 = os.listdir(folder2)

    # Iterate over subfolders in both folders
    for subfolder in subfolders1:
        subfolder1_path = os.path.join(folder1, subfolder)
        subfolder2_path = os.path.join(folder2, subfolder)

        if os.path.isdir(subfolder1_path) and os.path.isdir(subfolder2_path):
            images1 = set(os.listdir(subfolder1_path))
            images2 = set(os.listdir(subfolder2_path))

            # Determine images in folder2 that are not in folder1
            images_to_remove = images2 - images1

            # Remove images from folder2 that are not in folder1
            for image in images_to_remove:
                image_path = os.path.join(subfolder2_path, image)
                if os.path.isfile(image_path):
                    os.remove(image_path)
                    print(f"Removed: {image_path}")


if __name__ == "__main__":
    folder1 = "./data_42"  # Update with the path to your first folder
    folder2 = "./data"  # Update with the path to your second folder

    compare_and_remove(folder1, folder2)
