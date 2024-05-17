import os
import cv2
import numpy as np

# Full path
input_folder = input("Source Images Folder: ") 
output_folder = input("Processed Images Folder: ")

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all image files in the input folder
for file_name in os.listdir(input_folder):
    # Read the image file
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path)

    # Check if image was read successfully
    if image is None:
        print(f"Error: could not read {image_path}")
        continue

   
    flipped_image = cv2.flip(image, 0)
    
    # Save the rotated image to the output folder
    output_path = os.path.join(output_folder, 'cvcc' + file_name)
    cv2.imwrite(output_path, flipped_image)
