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

    # Resize the image to 222x22 without modifying it
    height, width, _ = image.shape
    if height >= width:
        resized_image = cv2.resize(image, (222, int(222 * height / width)))
    else:
        resized_image = cv2.resize(image, (int(222 * width / height), 222))
    center_x, center_y = resized_image.shape[1] // 2, resized_image.shape[0] // 2
    left_x, right_x = center_x - 111, center_x + 111
    top_y, bottom_y = center_y - 111, center_y + 111
    cropped_image = resized_image[top_y:bottom_y, left_x:right_x]

    # Create a black border around the cropped image
    bordered_image = cv2.copyMakeBorder(cropped_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
    
    # Adjust image brightness by 20%
    alpha = 1.5  # Increase brightness by 20% (1 + 0.2)
    beta = 0  # No bias
    adjusted_image = cv2.addWeighted(bordered_image, alpha, np.zeros(bordered_image.shape, dtype=np.uint8), 0, beta)

    # Save the adjusted image to the output folder
    output_path = os.path.join(output_folder, '1lvcb' + file_name)
    cv2.imwrite(output_path, adjusted_image)
