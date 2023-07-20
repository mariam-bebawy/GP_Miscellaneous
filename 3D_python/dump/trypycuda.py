import numpy as np
import scipy.ndimage as sciimg
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
import json

with open(os.path.join(os.path.dirname(__file__), 'trilinear_interpolation.cu')) as f:
    read_data = f.read()
f.closed

mod = SourceModule(read_data)
trilinear_interpolation_kernel = mod.get_function('trilinear_interpolation')
vol = np.random.rand(60*60*60).astype(np.float32)

vol_gpu=gpuarray.to_gpu(vol)
output_gpu = gpuarray.GPUArray(vol_gpu.shape, vol_gpu.dtype)
x=np.linspace(0,59,60)
y=np.linspace(0,59,60)
z=np.linspace(0,59,60)
 
X, Y, Z = np.meshgrid(x,y,z)
X_gpu=gpuarray.to_gpu(X.flatten().astype(np.float32))
Y_gpu=gpuarray.to_gpu(Y.flatten().astype(np.float32))
Z_gpu=gpuarray.to_gpu(Z.flatten().astype(np.float32))
#print(X,Y,Z)


block = (512,1,1)


#cuda.memcpy_htod(self._invG_gpu, invG)
#cuda.memcpy_htod(self._weights_gpu, G_half)
#(int n,float *x, float *y, float *z,float *intepolated_flow,float *vol)
N=60*60*60
#N_gpu=cuda.mem_alloc(N)
#cuda.memcpy_htod(N_gpu,N)

DIM=np.array([60,60,60])
DIM_GPU=gpuarray.to_gpu(DIM)


trilinear_interpolation_kernel(
                             np.int32(N),
                             DIM_GPU,
                             X_gpu,
                             Y_gpu,
                             Z_gpu,
                             output_gpu,
                             vol_gpu,
                             block=(1024,1,1), grid=(512,1))

print(vol_gpu[-10:-1])
print(output_gpu[-10:-1])




