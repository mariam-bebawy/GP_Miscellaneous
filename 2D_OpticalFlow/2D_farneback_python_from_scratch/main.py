import cv2 as cv
import numpy as np
from calc_optical_flow_farneback import calc_optical_flow_farneback

## set video path
video_path='videos/pnas.2109838119.sm01.mp4'
cap = cv.VideoCapture(video_path)
## read frame , ret is a boolean to say if frame is available
ret, frame1 = cap.read()
print(frame1.shape)
frame2=[]
##converte to gray scale
frame1_gray=cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# Creates a mask image with zeros for hsv visualization
mask = np.zeros_like(frame1)
# Saturation for mask image is set to 255
mask[..., 1] = 255

# While loop is implemented to run each and every frame in a loop
while(cap.isOpened()):
  
  # Read the current frame from the video and store it in frame variable
  ret, frame2 = cap.read()
  
  # New window is opened and Input frame is displayed
  cv.imshow("input", frame2)
  
  # Each frame is converted into grayscale
  frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
  
  # optical flow using Farneback method
  optical_flow = calc_optical_flow_farneback(frame1_gray, frame2_gray)
  
  # Magnitude and Angle of 2D vector is calculated
  mag, ang = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
  
  # Determines mask value using normalized magnitude
  mask[..., 0] = ang * 180 / np.pi / 2
  mask[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
  
  # Converts HSV (Hue Saturation Value) to RGB (or BGR)

  rgb_frame = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
  
  # Displays new output frame
  cv.imshow("optical flow", rgb_frame)
  
  
  frame1_gray = frame2_gray
  
  # Every frame is updated in period of 1 millisecond. When user presses "k" key, the window gets terminated
  if cv.waitKey(1) & 0xFF == ord('k'):
    break
# captured frame is released and display window is terminated after video frame ends
cap.release()
cv.destroyAllWindows()





