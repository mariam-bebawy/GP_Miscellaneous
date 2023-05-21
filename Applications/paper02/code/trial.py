import cv2
import numpy as np

position = (100, 100)
templateSize = 30
regionSize = 30
std = templateSize // 2

################################################################
################################################################
################################################################


def reduceToROI(img, pos, size):
    x, y = pos
    h, w = img.shape
    x_min = max(0, x - size // 2)
    x_max = min(w - 1, x + size // 2)
    y_min = max(0, y - size // 2)
    y_max = min(h - 1, y + size // 2)
    return img[y_min:y_max, x_min:x_max]


################################################################
################################################################
################################################################


frames, flow = [], []
path = "/path/to/my/video/file.mp4"
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        frames.append(img)
video = np.stack(frames, axis=0) # dimensions (T, H, W, C)

ref = reduceToROI(video[0], position, regionSize)
for i in video:
    # extract ROI from each frame
    next = reduceToROI(video[i])

    # calculate optical flow using farneback
    flowVec = cv2.calcOpticalFlowFarneback(ref, next, None)