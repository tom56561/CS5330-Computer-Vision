# Po-Shen Lee
# Date: 09/24/2024
# AL6: Adaptive Histogram Equalization for the entire image

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a dark and light image
Dark_img = np.random.randint(50, size=(500, 500), dtype=np.uint8)
Light_img = np.random.randint(low=205, high=255, size=(500, 500), dtype=np.uint8)

# Concatenate dark and light images
Dark_Light = np.concatenate((Dark_img, Light_img), axis=1)

# Apply adaptive histogram equalization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
adaptive_img = clahe.apply(Dark_Light)

# Display the original and equalized images
cv2.imshow('Original Image', Dark_Light)
cv2.imshow('Adaptive Equalized Image', adaptive_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
