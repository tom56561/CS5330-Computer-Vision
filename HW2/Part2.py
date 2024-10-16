# Po-Shen Lee
# Date: 10/16/2024
# HW2 - Part 2: Sobel and Canny Edge Detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('dog.jpeg', cv2.IMREAD_GRAYSCALE)

# Define Sobel filters
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Apply Sobel filters
def sobel_edge_detection(img, cutoff):
    grad_x = cv2.filter2D(img, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, sobel_y)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    _, edge_detected = cv2.threshold(gradient_magnitude, cutoff, 255, cv2.THRESH_BINARY)
    return edge_detected.astype(np.uint8)

# Apply filters to original image
sobel_50 = sobel_edge_detection(image, 50)
sobel_150 = sobel_edge_detection(image, 150)
canny = cv2.Canny(image, 100, 100)

# Apply filters to blurred image
blurred = cv2.GaussianBlur(image, (5, 5), 0)
sobel_blurred_50 = sobel_edge_detection(blurred, 50)
sobel_blurred_150 = sobel_edge_detection(blurred, 150)
canny_blurred = cv2.Canny(blurred, 100, 100)

# Display results in a 2x4 grid
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(2, 4, 2)
plt.imshow(sobel_50, cmap='gray')
plt.title("Sobel Cutoff 50")

plt.subplot(2, 4, 3)
plt.imshow(sobel_150, cmap='gray')
plt.title("Sobel Cutoff 150")

plt.subplot(2, 4, 4)
plt.imshow(canny, cmap='gray')
plt.title("Canny Edge")

plt.subplot(2, 4, 5)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred")

plt.subplot(2, 4, 6)
plt.imshow(sobel_blurred_50, cmap='gray')
plt.title("Sobel Blurred Cutoff 50")

plt.subplot(2, 4, 7)
plt.imshow(sobel_blurred_150, cmap='gray')
plt.title("Sobel Blurred Cutoff 150")

plt.subplot(2, 4, 8)
plt.imshow(canny_blurred, cmap='gray')
plt.title("Canny Blurred")

plt.tight_layout()
plt.show()

# What did you notice when you went from a lower threshold value to a higher one?
# Answer: A higher threshold value removes weaker edges, so fewer details appear in the Sobel result.

# What did you notice before and after applying a Gaussian Blur to the image?
# Answer: Blurring the image before applying edge detection reduces noise and helps to focus on prominent edges.
