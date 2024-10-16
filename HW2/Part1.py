# Po-Shen Lee
# Date: 10/16/2024
# HW2 - Part 1: 3D Visualization of Blurring

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('dog.jpeg', cv2.IMREAD_GRAYSCALE)

# Create a meshgrid for plotting in 3D
height, width = image.shape
x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)

# Apply Gaussian Blurs
blur_5x5 = cv2.GaussianBlur(image, (5, 5), 0)
blur_11x11 = cv2.GaussianBlur(image, (11, 11), 0)

# Plotting the original and blurred images in 3D
fig = plt.figure(figsize=(12, 4))

# Original image in 3D
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, image, cmap='gray')
ax1.set_title('Original Image')

# 5x5 Gaussian Blur in 3D
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, blur_5x5, cmap='gray')
ax2.set_title('5x5 Gaussian Blur')

# 11x11 Gaussian Blur in 3D
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, blur_11x11, cmap='gray')
ax3.set_title('11x11 Gaussian Blur')

plt.tight_layout()
plt.show()

# What do you notice about the 3D graphs as the filter size increases?
# Answer: As the filter size increases, the 3D graph becomes smoother and the details become less pronounced.
# This is because the Gaussian blur smooths out the intensity variations in the image.


