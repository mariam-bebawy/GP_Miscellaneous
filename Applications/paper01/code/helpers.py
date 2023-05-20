# importing libraries
import cv2, pickle, time
import numpy as np


################################################################
################################################################
################################################################

def saveVID(var, path, fps=15, flag=True):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"xvid"), fps, (var.shape[2], var.shape[1]), isColor=flag)
    for i in range(var.shape[0]):
        data = var[i, :, :]
        out.write(data.astype(np.uint8))
    out.release()

def showVID(path, title):
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        cv2.imshow(title, frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

def loadPickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file

def opticalFlowVanilla():
    return

def showFlow(img, flow, step):
    h, w = img.shape[:2]
    y, x = np.mgrid()
    return