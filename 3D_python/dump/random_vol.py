import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import os
import imageio
from tqdm import tqdm
import gzip
import pickle as pk
from Farneback_3dOpticalFlow import Farneback_3d
from Visualizer import visualizer 
import scipy.ndimage 
import farneback3d
import pyvista as pv
import time
from random import Random
v1=np.zeros((100,100,100),np.float32)

print(v1.shape)
SEED=5
random=Random(x=SEED)
r=2
num_of_points=1000
for _ in range(num_of_points):
    x=(random.randint(a=5,b=95))
    y=(random.randint(a=5,b=95))
    z=(random.randint(a=5,b=95))
    v1[x-r:x+r,y-r:y+r,z-r:z+r]=100
print(np.shape(np.where(v1==100)))




movement_vector = [2, 0, 0]


flow_ground_truth = np.stack([movement_vector[2] * np.ones(v1.shape, np.float32), movement_vector[1] * np.ones(
    v1.shape, np.float32), movement_vector[0] * np.ones(v1.shape, np.float32)], 0)

v2 = farneback3d.warp_by_flow(v1, flow_ground_truth)

optflow = Farneback_3d(
                 pyr_scale=0.8,         # Scaling between multi-scale pyramid levels
                 levels=3,              # Number of multi-scale levels
                 num_iterations=4,      # Iterations on each multi-scale level
                 winsize=3,             # Window size for Gaussian filtering of polynomial coefficients
                 poly_n=5,              # Size of window for weighted least-square estimation of polynomial coefficients
                 poly_sigma=1.2,        # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
    )
flow = optflow.calc_flow(v1, v2)
flow=np.asarray([flow[0].get(),flow[1].get(),flow[2].get()])

print(flow.shape)
print(np.where(flow[0][:]!=0.))
print(np.where((v1-v2)!=0.))
def plot_flow():
    plotter = pv.Plotter()

    positions = np.where(np.reshape(flow[0], (100, 100, 100)) != 0 )

    # this has the shape of flow fields but with true or false indicating whether or not there is a flow there
    boolean_positions = [flow[0] != 0 ]

    """
    directions contain the flow values at the locations where boolean_positions = True
    its shape is again (3 , number of locations where flow != 0)
    """
    directions = [np.extract(boolean_positions ,flow[0]),
                np.extract(boolean_positions , flow[1]),
                np.extract(boolean_positions , flow[2])]

    plotter.reset_camera(bounds = (0 , 120 , 0 , 120 , 0 , 120))


    plotter.add_arrows(np.swapaxes(positions,0,1), np.swapaxes(directions,0,1))
    plotter.add_camera_orientation_widget()
    plotter.show()

def plot_vol(vol):
    plotter = pv.Plotter()
    point_cloud = np.where(vol!= 0)
    # swap axes because the plotter needs a shape of (n , 3)
    point_cloud = np.swapaxes(point_cloud , 0,1)

    # first iteration we need to add_points to the plotter
        # add our point cloud to the plotter
    plotter.add_points(point_cloud, render_points_as_spheres=True,
                    point_size=5.0)
        

    plotter.write_frame()
    # put some delay 
    time.sleep(0.2)


plot_flow()
#plot_vol(v2)