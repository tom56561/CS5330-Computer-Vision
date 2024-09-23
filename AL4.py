# Eddie Lee
# Date: 09/17/2024
# Histogram Equalization for Colored Image

import cv2
import numpy as np

# Load the image
image = cv2.imread('images/flowers.jpg')

# Split the image into its three color channels (B, G, R)
blue_channel, green_channel, red_channel = cv2.split(image)

# Apply histogram equalization to each channel
equalized_blue = cv2.equalizeHist(blue_channel)
equalized_green = cv2.equalizeHist(green_channel)
equalized_red = cv2.equalizeHist(red_channel)

# Merge the equalized channels back into a color image
equalized_image = cv2.merge([equalized_blue, equalized_green, equalized_red])

# Display the original image
cv2.imshow('Original Image', image)

# Display the equalized image
cv2.imshow('Equalized Image', equalized_image)

# Save the equalized image as a file if needed
cv2.imwrite('images/equalized_flowers.jpg', equalized_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
