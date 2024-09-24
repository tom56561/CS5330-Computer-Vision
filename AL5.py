# Po-Shen Lee
# Date: 09/24/2024
# AL5: Display image in 3D with different colormap

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('images/dog.jpeg', 0)

# Get the dimensions of the image
height, width = img.shape

# Create coordinate arrays for the 3D plot
x = np.linspace(0, width, width, dtype=int)
y = np.linspace(0, height, height, dtype=int)
X, Y = np.meshgrid(x, y)

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface with a different colormap (e.g., viridis)
surf = ax.plot_surface(X, Y, img, cmap=plt.cm.viridis)

# Show the plot
plt.show()
