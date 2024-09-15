# Po-Shen Lee
# Date: 09/15/2024
# Lab 1 - Grey Scale

import cv2

# Load the original image
image = cv2.imread('images/flowers.jpg')

# Display the original image in a window
cv2.imshow('Original Image', image)

# Convert the image to grayscale
gray_image = cv2.imread('images/flowers.jpg', cv2.IMREAD_GRAYSCALE)

# Display the grayscale image in a second window
cv2.imshow('Grayscale Image', gray_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
