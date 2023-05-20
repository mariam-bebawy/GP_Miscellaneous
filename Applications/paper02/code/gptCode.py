import cv2
import numpy as np

from helpers import reduceToROI, calculateOpticalFlow, calculateAverageVelocity, isCloseToEdge, adjustRegionSize, generateWeightingMap, templateMatching, magnitude, distance, terminationConditionMet

from globals import stemplate, NeighborhoodSize, PyramidScale, NumPyramidLevels, FilterSize, MaxVelocityMagnitude, MaxPositionDifference, image, position, velocity, regionSize

while True:
    # Reduce image to region of interest
    roi = reduceToROI(image, position, regionSize)

    # Calculate optical flow
    flow = calculateOpticalFlow(roi)

    # Estimate velocity
    velocity = calculateAverageVelocity(flow, position, stemplate)

    # Adjust moving region if necessary
    if isCloseToEdge(position, regionSize):
        regionSize = adjustRegionSize(position, regionSize)

    # Generate weighting map
    weightingMap = generateWeightingMap(position, velocity)

    # Perform template matching to get peak correlation position
    correlationPosition = templateMatching(roi, weightingMap)

    # Check for large velocity and position difference
    if magnitude(velocity) > MaxVelocityMagnitude and distance(position, correlationPosition) > MaxPositionDifference:
        velocity = 0

    # Update position
    position = (position[0] + velocity, position[1] + velocity)

    # Check termination condition
    if terminationConditionMet():
        break

# Final position estimation
estimatedPosition = position

print("Estimated position:", estimatedPosition)
