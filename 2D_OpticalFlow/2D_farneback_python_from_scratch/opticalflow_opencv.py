# Importing cv2 and numpy library
import cv2 as cv
import numpy as np
# Read the video input
cap = cv.VideoCapture('videos/pnas.2109838119.sm01.mp4')
# read the first captured frame of the video and stored in first_cap_frame
ret, first_cap_frame = cap.read()
# BGR frame is converted into grayscale 
first_gray = cv.cvtColor(first_cap_frame, cv.COLOR_BGR2GRAY)
# Creates a mask image with zeros
mask = np.zeros_like(first_cap_frame)
# Saturation for mask image is set to 255
mask[..., 1] = 255
# While loop is implemented to run each and every frame in a loop
while(cap.isOpened()):
  
  # Read the current frame from the video and store it in frame variable
  ret, frame = cap.read()
  
  # New window is opened and Input frame is displayed
  cv.imshow("input", frame)
  
  # Each frame is converted into grayscale
  current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  
  # optical flow using Farneback method
  optical_flow = cv.calcOpticalFlowFarneback(first_gray, current_gray,
                  None,
                  0.2, 4, 12, 2, 3, 1.1, 0)
  
  # Magnitude and Angle of 2D vector is calculated
  mag, ang = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
  
  # Determines mask value using normalized magnitude
  mask[..., 0] = ang * 180 / np.pi / 2
  
  
  mask[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
  
  # Converts HSV (Hue Saturation Value) to RGB (or BGR)
  rgb_frame = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
  
  # Displays new output frame
  cv.imshow("optical flow", rgb_frame)
  
  
  first_gray = current_gray
  
  # Every frame is updated in period of 1 millisecond. When user presses "k" key, the window gets terminated
  if cv.waitKey(1) & 0xFF == ord('k'):
    break
# captured frame is released and display window is terminated after video frame ends
cap.release()
cv.destroyAllWindows()