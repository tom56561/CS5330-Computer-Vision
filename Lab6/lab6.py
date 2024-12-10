# Po-Shen Lee
# Date: 10/26/2024
# Lab 6 - Improved Contours and Shadows Detection

import cv2
import numpy as np

# Load the image
image = cv2.imread('blocks.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Step 2: Apply Adaptive Thresholding with modified parameters
# Try a smaller block size and adjust 'C' to control the sensitivity
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 3)

# Step 3: Use a larger kernel for morphological operations
# Adjusting the kernel size for better shadow reduction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

# Step 4: Find contours
contours, _ = cv2.findContours(
    morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Draw contours and calculate the center of each contour
for contour in contours:
    # Calculate the contour area
    area = cv2.contourArea(contour)
    if area > 500:  # Adjust the area threshold as needed to filter out noise
        # Draw contour
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Mark the center point
        cv2.circle(image, (cX, cY), 5, (255, 0, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Display the result
cv2.imshow('Contour and Center Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result if needed
cv2.imwrite('final.jpg', image)
