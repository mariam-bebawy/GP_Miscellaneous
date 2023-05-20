import numpy as np
import cv2, time

path = "./outputs/VIDsegCT.avi"

def testing(path):
    cap = cv2.VideoCapture(path)
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print(prev_gray.shape)
    mask = np.zeros_like(first_frame)
    print(mask.shape)
    mask[..., 1] = 255

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        cv2.imshow("input segmented lung CT", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 5, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        cv2.imshow("dense optical flow", rgb)

        prev = gray
        if cv2.waitKey(25) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

testing(path)