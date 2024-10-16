# Po-Shen Lee
# Date: 10/15/2024
# Lab 5 - Store Brand Night Vision with Dark Gold Tone

import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('images/friends.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: Apply a Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Apply edge detection using the Canny algorithm
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Step 3: Enhance the contrast using histogram equalization
equalized = cv2.equalizeHist(image)

# Step 4: Combine the edges with the original image to highlight important elements
combined = cv2.addWeighted(equalized, 0.5, edges, 0.5, 0)

# Step 5: Apply a color map to simulate a night vision effect (green tint)
night_vision = cv2.applyColorMap(combined, cv2.COLORMAP_BONE)

# Step 6: Enhance brightness to make important elements stand out more
alpha = 1.5  # Contrast control
beta = 30    # Brightness control
enhanced_night_vision = cv2.convertScaleAbs(
    night_vision, alpha=alpha, beta=beta)

# Display night vision effect images
cv2.imshow('Night Vision Effect', enhanced_night_vision)

cv2.waitKey(0)
cv2.destroyAllWindows
