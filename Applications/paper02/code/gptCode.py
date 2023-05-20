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

def reduceToROI(image, position, size):
    x, y = position
    h, w = image.shape
    x_min = max(0, x - size // 2)
    x_max = min(w - 1, x + size // 2)
    y_min = max(0, y - size // 2)
    y_max = min(h - 1, y + size // 2)
    return image[y_min:y_max, x_min:x_max]

def calculateOpticalFlow(roi):
    prev_frame = roi[0]
    flow = np.zeros_like(roi)

    for i in range(1, len(roi)):
        curr_frame = roi[i]
        flow_vectors = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, PyramidScale, NumPyramidLevels, FilterSize, NeighborhoodSize, 5, 1.2, 0)
        flow[i] = flow_vectors[..., 0]  # Extract x-component of flow vectors
        prev_frame = curr_frame

    return flow

def calculateAverageVelocity(flow, position, size):
    x, y = position
    x_min = max(0, x - size // 2)
    x_max = min(flow.shape[2] - 1, x + size // 2)
    y_min = max(0, y - size // 2)
    y_max = min(flow.shape[1] - 1, y + size // 2)
    region = flow[:, y_min:y_max, x_min:x_max]
    return np.mean(region)

def isCloseToEdge(position, size):
    x, y = position
    h, w = image.shape
    return x - size // 2 < 0 or x + size // 2 >= w or y - size // 2 < 0 or y + size // 2 >= h

def adjustRegionSize(position, size):
    x, y = position
    h, w = image.shape
    x_min = max(0, x - size // 2)
    x_max = min(w - 1, x + size // 2)
    y_min = max(0, y - size // 2)
    y_max = min(h - 1, y + size // 2)
    new_size = max(x_max - x_min, y_max - y_min)
    return new_size

def generateWeightingMap(position, velocity):
    x, y = position
    sigma_vel = np.std(velocity)
    weighting_map = np.zeros_like(image)
    weighting_map[y, x] = 1  # Set the pixel at the predicted position to 1
    weighting_map = cv2.GaussianBlur(weighting_map, (stemplate, stemplate), sigma_vel)
    return weighting_map

def templateMatching(roi, weighting_map):
    result = cv2.matchTemplate(roi, weighting_map, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    return x, y

def magnitude(vector):
    return np.linalg.norm(vector)

def distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def terminationConditionMet():
    # Implement your termination condition here
    return False

# Load image
image = cv2.imread("path/to/image.jpg", 0)  # Grayscale image

# Initialize variables
position = (100, 100)
velocity = 0
regionSize = stemplate

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
