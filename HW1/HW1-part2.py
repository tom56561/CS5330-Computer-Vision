# Po-Shen Lee
# Date: 09/29/2024
# Refactored HW1 - Part 2: Image Coloring (Thermal Vision)

import cv2
import numpy as np


def thermal_coloring(image):
    # Normalize the grayscale image to [0, 255] range
    norm_image = cv2.normalize(
        image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Create a thermal color image using a custom color map (closer to OpenCV style)
    thermal_image = np.zeros(
        (image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Apply the thermal color map
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            brightness = norm_image[i, j]

            # Apply a more graduated color scheme for thermal effect
            if brightness < 85:  # Darker values -> Blue to Cyan
                thermal_image[i, j] = [255, int(3 * brightness), 0]
            elif brightness < 170:  # Mid-range -> Cyan to Yellow
                brightness_adjusted = brightness - 85
                thermal_image[i, j] = [
                    255 - int(3 * brightness_adjusted), 255, 0]
            else:  # Brighter values -> Yellow to Red
                brightness_adjusted = brightness - 170
                thermal_image[i, j] = [
                    0, 255 - int(3 * brightness_adjusted), 255]

    return thermal_image


# Load the image from canvas
image = cv2.imread('flowers.jpg', cv2.IMREAD_GRAYSCALE)

# Apply custom thermal coloring
thermal_image_custom = thermal_coloring(image)

# Display the custom thermal image
cv2.imshow('Thermal Image', thermal_image_custom)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the custom thermal image
cv2.imwrite('thermal_flowers.jpg', thermal_image_custom)
