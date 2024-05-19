from PIL import Image
import os

def create_collage(images, output_path):
    # Calculate dimensions for the collage
    num_rows = 2
    num_cols = 2
    image_width = 600  # Adjust this to your preference
    image_height = 600  # Adjust this to your preference
    spacing = 10  # Adjust this to your preference
    collage_width = (image_width + spacing) * num_cols
    collage_height = (image_height + spacing) * num_rows

    # Create a new blank image for the collage
    collage = Image.new('RGB', (collage_width, collage_height), color='white')

    # Paste images into the collage
    for i, img_path in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * (image_width + spacing)
        y = row * (image_height + spacing)
        img = Image.open(img_path)
        img = img.resize((image_width, image_height))
        collage.paste(img, (x, y))

    # Save the collage
    collage.save(output_path)
    collage.show()

def main():
    input_folder = "./output_images/Words"
    output_folder = "collage_output/"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of image files in the input folder
    image_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)
                   if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Sort image files in ascending order
    image_files.sort()

    print(f"Found {len(image_files)} images.")

    # Define the number of rows and columns for the collage
    num_rows = 2
    num_cols = 2

    # Calculate the total number of collages expected to be created
    total_collages = (len(image_files) + (num_rows * num_cols - 1)) // (num_rows * num_cols)
    print(f"Total collages expected: {total_collages}")

    # Create collages until all images are processed
    collage_count = 1
    while image_files:
        collage_images = image_files[:num_cols * num_rows]  # Take enough images for each collage
        collage_output_path = os.path.join(output_folder, f"words_{collage_count}.jpg")
        create_collage(collage_images, collage_output_path)
        print(f"Collage {collage_count} created and saved as {collage_output_path}")
        image_files = image_files[num_cols * num_rows:]  # Move to the next set of images
        collage_count += 1

if __name__ == "__main__":
    main()
