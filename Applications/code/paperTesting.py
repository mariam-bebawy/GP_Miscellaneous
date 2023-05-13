import os, sys, cv2
import numpy as np
import slicing, helpers
# import testFile

# STEPS AGAIN
# 1. preprocessing ==> already done
# 2. slice selection
# 3. optical flow method selection
# 4. post processing


################################################################
################################################################
################################################################

# segCT = helpers.loadPickle("./outputsPickle/segCT.pickle")
segCT = helpers.loadPickle("./outputsPickle/CT.pickle")
print(f"segmented CT scane shape = {segCT.shape}")
frames, height, width = segCT.shape

ref = segCT[0]
mask = np.zeros((height, width, 3))
print(mask.shape)
mask[..., 1] = 255

# ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
colors = []
for i in range(0, frames):
    curr = segCT[i]
    flow = cv2.calcOpticalFlowFarneback(ref, curr, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # apparently mask ==> CV_64F // need it as uint8
    mask = mask.astype(np.uint8)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    colors.append(rgb)
    ref = curr
colors = np.array(colors)
print(f"done with optical flow")

# helpers.saveVID(colors, "./outputsFlow/flow_segCT.avi")
helpers.saveVID(colors, "./outputsFlow/flow_CT.avi")
print(f"done saving flow video")

# helpers.showVID("./outputsFlow/flow_segCT.avi", "farneback flow")
helpers.showVID("./outputsFlow/flow_CT.avi", "farneback flow")
print(f"done showing video")


################################################################
################################################################
################################################################
