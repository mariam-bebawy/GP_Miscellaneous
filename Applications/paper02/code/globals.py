import cv2
import numpy as np

# Define parameters
stemplate = 3
NeighborhoodSize = 5
PyramidScale = 0.5
NumPyramidLevels = 3
FilterSize = 5
MaxVelocityMagnitude = 5
MaxPositionDifference = 20

# Load image
image = cv2.imread("path/to/image.jpg", 0)  # Grayscale image

# Initialize variables
position = (100, 100)
velocity = 0
regionSize = stemplate
