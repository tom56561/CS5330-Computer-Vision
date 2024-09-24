# Po-Shen Lee
# Date: 09/17/2024
# Grayscale and Inversion without built-in cv2 functions

import cv2
import numpy as np

# Load the image as a grayscale image
gray_image = cv2.imread('images/dog.jpeg', cv2.IMREAD_GRAYSCALE)

# Invert the grayscale image (255 - pixel value)
inverted_image = 255 - gray_image

# Display the original grayscale image
cv2.imshow('Grayscale Image', gray_image)

# Display the inverted image
cv2.imshow('Inverted Image', inverted_image)

# Save the inverted image as a file if needed
cv2.imwrite('inverted_dog.jpeg', inverted_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
