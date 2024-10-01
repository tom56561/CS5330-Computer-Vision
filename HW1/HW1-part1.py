# Po-Shen Lee
# Date: 09/29/2024
# HW1 - Part 1: Histogram Equalization

import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    # Flatten the image to 1D array
    flat = image.flatten()

    # Calculate the histogram (count of each intensity value)
    hist = np.bincount(flat, minlength=256)

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(hist)

    # Normalize CDF to [0, 255]
    cdf_normalized = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # Map the old pixel values to the new ones using the CDF
    equalized_image = cdf_normalized[flat]
    equalized_image = equalized_image.reshape(image.shape)

    return equalized_image


# Load the dark image from canvas
image = cv2.imread('dark_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply manual histogram equalization
equalized_image_manual = histogram_equalization(image)

# Compare with OpenCV's built-in equalizeHist function
equalized_image_cv2 = cv2.equalizeHist(image)

# Display results
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(equalized_image_manual,
                                 cmap='gray'), plt.title('Manual Equalization')
plt.subplot(1, 3, 3), plt.imshow(equalized_image_cv2,
                                 cmap='gray'), plt.title('OpenCV Equalization')
plt.show()

# Save the images if needed
cv2.imwrite('equalized_manual.jpg', equalized_image_manual)
cv2.imwrite('equalized_cv2.jpg', equalized_image_cv2)
