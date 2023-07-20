import imageio
import numpy as np
import matplotlib.pyplot as plt
import farneback3d
import open3d as o3d
import cv2

im = imageio.volread(r'D:\GP\3D ISA\videos\traj_chr10_chr11_100.tiff')
imarray = np.array(im)
print(imarray.shape)
points=imarray[80]
print(points.shape)
from mayavi.mlab import *

from mayavi import mlab

#contour3d(imarray[80],color=(0,0,1))
#contour3d(imarray[81],color=(1,0,0))



# mlab.show()
optflow = farneback3d.Farneback(
        pyr_scale=0.8,         # Scaling between multi-scale pyramid levels
        levels=5,              # Number of multi-scale levels
        num_iterations=5,      # Iterations on each multi-scale level
        winsize=3,             # Window size for Gaussian filtering of polynomial coefficients
        poly_n=3,              # Size of window for weighted least-square estimation of polynomial coefficients
        poly_sigma=1,          # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
    )
#optflow = farneback3d.Farneback()

vol0=imarray[80].astype(np.float32)
vol1=imarray[81].astype(np.float32)
#print(np.unique(vol0))
#print(np.where(vol0==100.))
#print(np.where(vol1==100.))
movement_vector = [0, 20, 0]

vol0 = np.zeros([60, 60, 60], np.float32)
vol0[30:31, 30:31, 30:31] = 100.
flow_ground_truth = 4* np.ones([*vol0.shape,3], np.float32)
flow_ground_truth = np.stack([movement_vector[2] * np.ones(vol0.shape, np.float32), movement_vector[1] * np.ones(vol0.shape, np.float32), movement_vector[0] * np.ones(vol0.shape, np.float32)], 0)

vol1 = farneback3d.warp_by_flow(vol0, flow_ground_truth)


flow = optflow.calc_flow(vol0, vol1)
#flow = optflow.calc_flow(vol0[50:100,50:100,50:100], vol1[50:100,50:100,50:100])
#flow = cv2.calcOpticalFlowFarneback(vol0[:,:,50], vol1[:,:,50],None, 0.5, 3, 15, 3, 5, 1.2, 0)

print(flow.shape)
print(np.unique(flow).shape)

#mask=vol1+vol0
mask=1

#contour3d(vol0,color=(0,0,1))
#contour3d(vol1,color=(1,0,0))
quiver3d(flow[2],flow[1],flow[0],color=(0,1,0))
mlab.show()
