import cv2
import numpy as np

from globals import stemplate, NeighborhoodSize, PyramidScale, NumPyramidLevels, FilterSize, MaxVelocityMagnitude, MaxPositionDifference, image, position, velocity, regionSize

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
