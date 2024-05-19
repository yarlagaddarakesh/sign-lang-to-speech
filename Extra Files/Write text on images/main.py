import cv2
import os
from PIL import Image, ImageDraw, ImageFont

def write_text_on_image(image_path, text, output_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert image to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to PIL Image
    pil_image = Image.fromarray(image)
    
    # Get a font
    font = ImageFont.load_default()
    
    # Set text size to 20 pixels
    font = ImageFont.truetype("arial.ttf", 35)
    
    # Get a drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Position text at top-left corner
    text_position = (10, 10)  # Adjust the values to your preference
    
    # Draw text on image
    draw.text(text_position, text, fill="black", font=font)
    
    # Save image with text
    pil_image.save(output_path)

def extract_text_from_image_name(image_name):
    # Remove file extension
    image_name = os.path.splitext(image_name)[0]
    
    # Extract text from image name (assuming text is separated by underscores)
    text = " ".join(image_name.split('_'))
    
    return text

def main():
    input_folder = "../Datasets/New folder/Words"
    output_folder = "output_images/Words/"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Extract text from image name
            text = extract_text_from_image_name(filename)
            
            # Write text on image and save it
            write_text_on_image(image_path, text, output_path)
            print(f"Text '{text}' written on {filename} and saved as {output_path}")

if __name__ == "__main__":
    main()
