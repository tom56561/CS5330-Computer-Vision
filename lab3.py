# Po-Shen Lee
# Date: 09/29/2024
# Lab 3 - Filtering Scale

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image, noise_level):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(noise_level * total_pixels)

    # Add salt noise (white pixels)
    for i in range(num_salt):
        y_coord = random.randint(0, image.shape[0] - 1)
        x_coord = random.randint(0, image.shape[1] - 1)
        noisy_image[y_coord, x_coord] = 255

    # Add pepper noise (black pixels)
    for i in range(num_salt):
        y_coord = random.randint(0, image.shape[0] - 1)
        x_coord = random.randint(0, image.shape[1] - 1)
        noisy_image[y_coord, x_coord] = 0

    return noisy_image

# Function to apply a filter with zero padding
def apply_filter(image, kernel_size):
    return cv2.filter2D(image, -1, np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2))


# Load the image and convert it to grayscale
image = cv2.imread('images/dog.jpeg', cv2.IMREAD_GRAYSCALE)

# Add different levels of salt and pepper noise
noise_1 = add_salt_and_pepper_noise(image, 0.01)
noise_10 = add_salt_and_pepper_noise(image, 0.10)
noise_50 = add_salt_and_pepper_noise(image, 0.50)

# Apply 3x3 and 5x5 filters with zero padding
filtered_3x3_noise_1 = apply_filter(noise_1, 3)
filtered_3x3_noise_10 = apply_filter(noise_10, 3)
filtered_3x3_noise_50 = apply_filter(noise_50, 3)

filtered_5x5_noise_1 = apply_filter(noise_1, 5)
filtered_5x5_noise_10 = apply_filter(noise_10, 5)
filtered_5x5_noise_50 = apply_filter(noise_50, 5)

# Plotting the results in a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes[0, 0].imshow(noise_1, cmap='gray')
axes[0, 0].set_title('1% Noise')
axes[0, 1].imshow(noise_10, cmap='gray')
axes[0, 1].set_title('10% Noise')
axes[0, 2].imshow(noise_50, cmap='gray')
axes[0, 2].set_title('50% Noise')

axes[1, 0].imshow(filtered_3x3_noise_1, cmap='gray')
axes[1, 0].set_title('3x3 Filter on 1%')
axes[1, 1].imshow(filtered_3x3_noise_10, cmap='gray')
axes[1, 1].set_title('3x3 Filter on 10%')
axes[1, 2].imshow(filtered_3x3_noise_50, cmap='gray')
axes[1, 2].set_title('3x3 Filter on 50%')

axes[2, 0].imshow(filtered_5x5_noise_1, cmap='gray')
axes[2, 0].set_title('5x5 Filter on 1%')
axes[2, 1].imshow(filtered_5x5_noise_10, cmap='gray')
axes[2, 1].set_title('5x5 Filter on 10%')
axes[2, 2].imshow(filtered_5x5_noise_50, cmap='gray')
axes[2, 2].set_title('5x5 Filter on 50%')

plt.tight_layout()
plt.show()
