# Po-Shen Lee
# Date: 10/08/2024
# Lab 4 - Marilyn Einstein Image Processing

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image for low pass filter
image1 = cv2.imread('images/puppy.jpg', cv2.IMREAD_GRAYSCALE)
# Image for high pass filter
image2 = cv2.imread('images/tree.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: Apply a low pass filter (11x11 Gaussian blur) to image1
low_pass_image = cv2.GaussianBlur(image1, (11, 11), 0)

# Step 2: Create a high pass filter by subtracting the low-pass filtered version from the original image
low_pass_image2 = cv2.GaussianBlur(image2, (11, 11), 0)
high_pass_image = cv2.subtract(
    image2, low_pass_image2) + 127  # Adjust the brightness

# Step 3: Combine the low pass and high pass images
combined_image = cv2.addWeighted(low_pass_image, 0.5, high_pass_image, 0.5, 0)

# Display the images in a 3x1 grid
plt.figure(figsize=(12, 6))

# Low pass filtered image
plt.subplot(1, 3, 1)
plt.imshow(low_pass_image, cmap='gray')
plt.title('Low Pass Filtered Image')
plt.axis('off')

# High pass filtered image
plt.subplot(1, 3, 2)
plt.imshow(high_pass_image, cmap='gray')
plt.title('High Pass Filtered Image')
plt.axis('off')

# Combined image
plt.subplot(1, 3, 3)
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Image')
plt.axis('off')

plt.tight_layout()
plt.show()
