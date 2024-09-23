# Po-Shen Lee
# Date: 09/23/2024
# Lab 2 - Grey Scale

import cv2
import numpy as np

# Load the original image
image = cv2.imread('images/flowers.jpg')

# Average Method: (R + G + B) / 3
def average_method(img):
    return np.mean(img, axis=2).astype(np.uint8)

# NTSC Method: 0.299 * R + 0.587 * G + 0.114 * B
def ntsc_method(img):
    return (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype(np.uint8)

# Apply Average Method
gray_avg = average_method(image)

# Apply NTSC Method
gray_ntsc = ntsc_method(image)

# Compare with OpenCV built-in grayscale conversion
gray_cv2 = cv2.imread('images/flowers.jpg', cv2.IMREAD_GRAYSCALE)

# Display all images
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale (Average Method)', gray_avg)
cv2.imshow('Grayscale (NTSC Method)', gray_ntsc)
cv2.imshow('Grayscale (OpenCV)', gray_cv2)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the grayscale images if needed
cv2.imwrite('gray_avg_flowers.jpg', gray_avg)
cv2.imwrite('gray_ntsc_flowers.jpg', gray_ntsc)
