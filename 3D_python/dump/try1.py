import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import os
import imageio
from tqdm import tqdm
import gzip
import pickle as pk
from Farneback_3dOpticalFlow import Farneback_3d
#from Visualizer import visualizer 
import scipy.ndimage 
class mytiff:
  def __init__(self, path):
    self.volumes = imageio.imread(path)[:5]
    self.number_of_volumes,self.height,self.width,self.depth = np.shape(self.volumes)
    self.dim=np.array([self.height,self.width,self.depth])
    self.N=self.height*self.width*self.depth
    self.DIM_GPU=gpuarray.to_gpu(self.dim)
    self.flow_field =None
    self.position=None 
    self.poistions_folderpath=r"D:\GP\3D ISA\python\positions"
    self.fields_folderpath=r"D:\GP\3D ISA\python\fields"
    blocksize=int(1024)
    gridsize=int(np.ceil(self.N/blocksize))
    self.block = (blocksize,1,1)
    self.grid = (gridsize,1)

    with open(os.path.join(os.path.dirname(__file__), 'trilinear_interpolation.cu')) as f:
        read_data = f.read()
    f.closed

    mod = SourceModule(read_data)
    self.trilinear_interpolation_kernel = mod.get_function('trilinear_interpolation')
    self.optflow = Farneback_3d(
                 pyr_scale=0.8,         # Scaling between multi-scale pyramid levels
                 levels=3,              # Number of multi-scale levels
                 num_iterations=4,      # Iterations on each multi-scale level
                 winsize=3,             # Window size for Gaussian filtering of polynomial coefficients
                 poly_n=5,              # Size of window for weighted least-square estimation of polynomial coefficients
                 poly_sigma=1.2,        # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
    )
    print("file path =" ,path)
    print("number of time steps =", self.number_of_volumes)
    print("volume dimensions =", self.dim)



  def optical_flow(self):
    
    vol1 = self.volumes[0].astype(np.float32)
    #vol1=scipy.ndimage.binary_dilation(vol1).astype(np.float32)
    for i in tqdm(range(self.number_of_volumes-1)):
        vol2 = self.volumes[i+1].astype(np.float32)
        #vol2=scipy.ndimage.binary_dilation(vol2).astype(np.float32)
        output_vz, output_vy, output_vx = self.optflow.calculate_flow(vol1, vol2)

        with gzip.open(os.path.join(os.path.join(self.fields_folderpath,str(str(i))+".gz")), "wb") as f:
          pk.dump(np.asarray([output_vz, output_vy, output_vx]), f)
        if (i==0):
          self.__init_positions()
        self.__construct_trajectories(INDEX=i+1)
        vol1 = vol2

  def __init_positions(self):

    #x=np.linspace(0 , self.height - 1 , self.height)
    #y=np.linspace(0 , self.width - 1 , self.width)
    #z=np.linspace(0 , self.depth - 1 , self.depth)
    #X, Y, Z = np.meshgrid(x,y,z,indexing='ij')
    #X=X.flatten().astype(np.float32)
    #Y=Y.flatten().astype(np.float32)
    #Z=Z.flatten().astype(np.float32)
    
    points = np.where(np.reshape(self.volumes[0] , (self.height, self.width, self.depth)) != 0 )

    points=np.asarray(points,dtype=np.float32)
    #points=np.array([X,Y,Z])
    with gzip.open(os.path.join(self.poistions_folderpath,"0.gz"), "wb") as f:
      pk.dump(points, f)
    self.position = gpuarray.to_gpu(points)
    self.next_position=self.position.copy()
    self.N=points.shape[1]
    self.interpolated_flow = gpuarray.GPUArray(self.N, np.float32)



     

  def __construct_trajectories(self,INDEX):   
    for i in range(3):
        self.trilinear_interpolation_kernel(
                          np.int32(self.N),
                          self.DIM_GPU,
                          self.position[0],
                          self.position[1],
                          self.position[2],
                          self.interpolated_flow,
                          self.flow_field[i],
                          block=self.block,
                          grid = self.grid)
        self.next_position[i]=self.position[i]+self.interpolated_flow
    self.position=self.next_position
    with gzip.open(os.path.join(self.poistions_folderpath,str(str(INDEX)+".gz")), "wb") as f:
      pk.dump(self.next_position.get(), f)


 

tiff=mytiff(r'D:\GP\3D ISA\videos\heart_LV_chr01_02_100_comp.tiff')

tiff.optical_flow()


# vis = visualizer()
# vis.Trajectory_visualization()








