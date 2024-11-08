# Po-Shen Lee
# Date: 11/07/2024
# Lab 8 - Image Reading, Resizing, and Display with Titles

import os
import cv2
import re
import matplotlib.pyplot as plt

# Define the directory path to Lab8 data
data_dir = "/courses/CS5330.202510/data/Lab8"

# Lists to hold the images and their class names
images = []
class_names = ["cat", "dog", "squirrel"]

# Regular expression to capture the class name from the filename
# class_name_pattern = re.compile(r'^[a-zA-Z]+')

# Loop through each file in the Lab8 data directory
for filename in os.listdir(data_dir):
    print("Processing file:", filename)  # Debugging line
    if filename.endswith(".jpg"):  # Update to .jpg for this case
        # Extract the class name from the file name
        # match = class_name_pattern.match(filename)
        # if match:
        #     class_name = match.group(0)

        # Read in the image
        img_path = os.path.join(data_dir, filename)
        image = cv2.imread(img_path)

        if image is not None:
            # Resize the image to 78x78 pixels
            resized_image = cv2.resize(image, (78, 78))

            # Convert the image to RGB for displaying with matplotlib
            resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # Append the image and class name to the lists
            images.append(resized_image_rgb)
            # class_names.append(class_name)
        else:
            print("Warning: Failed to load image:", img_path)  # Debugging line

# Check if images list is populated
if len(images) == 0:
    print("No images were loaded. Please check the directory path and image formats.")
else:
    # Display the images in a subplot with their class names as titles
    fig, axs = plt.subplots(1, len(images), figsize=(10, 10))
    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.set_title(class_names[i])
        ax.axis("off")
    plt.show()
