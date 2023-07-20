import os
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import utils
from utils import divup
import filtering
import scipy.ndimage as sciimg

_NUM_POLY_COEFFICIENTS = 10
_MIN_VOL_SIZE = 32

class Farneback_3d:
    def __init__(self,
                 pyr_scale=0.9,
                 levels=15,
                 winsize=9,
                 num_iterations=5,
                 poly_n=5,
                 poly_sigma=1.2):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.num_iterations = num_iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self._resize_kernel_size_factor = 4
        self._max_resize_kernel_size = 9
        with open(os.path.join(os.path.dirname(__file__), 'farneback_kernels.cu')) as f:
            read_data = f.read()
        f.closed
        mod = SourceModule(read_data)
        self._update_matrices_kernel = mod.get_function(
            'FarnebackUpdateMatrices')
        self._invG_gpu = mod.get_global('invG')[0]
        self._weights_gpu = mod.get_global('weights')[0]
        self._poly_expansion_kernel = mod.get_function('calcPolyCoeficients')
        self._warp_kernel = mod.get_function('warpByFlowField')
        self._r1_texture = mod.get_texref('sourceTex')
        self._solve_equations_kernel = mod.get_function('solveEquationsCramer')



    def calc_flow(self,cur_vol: np.ndarray,
                  next_vol: np.ndarray):
        
        dim = len(cur_vol.shape)

        assert dim == 3, 'wrong dimension'
        assert cur_vol.shape == next_vol.shape
        assert cur_vol.dtype == np.float32, 'wrong dtype'
        assert next_vol.dtype == np.float32, 'wrong dtype'

        prev_flow_gpu = None
        imgs = [gpuarray.to_gpu(next_vol), gpuarray.to_gpu(cur_vol)]
        flow_gpu = None

        for k in range(self.levels, -1, -1):
            #print('Scale %i' % k)

            scale = self.pyr_scale**k

            if np.any(np.array(cur_vol.shape) * scale < _MIN_VOL_SIZE):
                continue

            sigma = (1. / scale - 1) * 0.5
            smooth_sz = int(sigma * self._resize_kernel_size_factor + 0.5) | 1
            smooth_sz = max(smooth_sz, 3)
            smooth_sz = min(smooth_sz, self._max_resize_kernel_size)

            scale_shape = [int(np.round(x * scale)) for x in cur_vol.shape]

            if prev_flow_gpu is not None:
                flow_gpu *= 1 / self.pyr_scale
                scaled_flow_gpu = gpuarray.GPUArray(
                    [3, *scale_shape], np.float32)
                for i in range(3):
                    self._resize(
                        flow_gpu[i], scaled_flow_gpu[i], dst_shape=scale_shape)

                flow_gpu = scaled_flow_gpu

            else:
                flow_gpu = gpuarray.zeros([dim, *scale_shape], np.float32)

            R = gpuarray.GPUArray(
                [2, _NUM_POLY_COEFFICIENTS - 1, *flow_gpu.shape[1:]], np.float32)
            M = gpuarray.GPUArray([9, *flow_gpu.shape[1:]], np.float32)

            for i in range(2):
                if k == 0:
                    I = imgs[i]
                else:
                    I = filtering.smooth_cuda_gauss(imgs[i], sigma,
                                                                 smooth_sz)
                    I = self._resize(I, scaling=scale)

                self._FarnebackPolyExp(I, R[i], self.poly_n, self.poly_sigma)

            # dsareco.visualization.imshow(R[0])
            # dsareco.visualization.imshow(R[1])

            self._FarnebackUpdateMatrices_gpu(R[0], R[1], flow_gpu, M)

            # dsareco.visualization.imshow(M,"M")

            for i in range(self.num_iterations):
            #print('iteration %i' % i)
                self._FarnebackUpdateFlow_GaussianBlur_gpu(
                    R[0], R[1], flow_gpu, M, self.winsize, i < self.num_iterations - 1)
               
            prev_flow_gpu = flow_gpu

        if flow_gpu is not None:
            return [flow_gpu[0].ravel(),flow_gpu[1].ravel(),flow_gpu[2].ravel()]
        else:
            return [gpuarray.zeros(cur_vol.shape, np.float32).ravel(),gpuarray.zeros(cur_vol.shape, np.float32).ravel(),gpuarray.zeros(cur_vol.shape, np.float32).ravel()]
        
    def _FarnebackPrepareGaussian(self, n, sigma):
        if sigma < 1e-7:
            sigma = n * 0.3

        x = np.arange(-n, n + 1, dtype=np.float32)
        g = np.exp(-(x * x) / (2 * sigma * sigma))
        g /= np.sum(g)
        # xg = x * g
        # xxg = x * xg

        G = np.zeros(
            (_NUM_POLY_COEFFICIENTS, _NUM_POLY_COEFFICIENTS), np.float32)

        G_half = np.zeros((10, 2 * n + 1, 2 * n + 1, 2 * n + 1), np.float32)

        # G:  sum_xyz weight_xyz *(1 x y z xx yy zz xy xz yz) * (1 x y z xx yy zz xy xz yz)^T

        for z in range(-n, n + 1):
            for y in range(-n, n + 1):
                for x in range(-n, n + 1):
                    gauss_weight = g[z + n] * g[y + n] * g[x + n]
                    base_vector = np.atleast_2d(np.array(
                        [1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z], np.float32))
                    matrix = np.matmul(np.transpose(base_vector), base_vector)
                    G += matrix * gauss_weight
                    G_half[:, z + n, y + n, x + n] = gauss_weight * base_vector

        invG = np.linalg.inv(G)

        return invG, G_half

    def _FarnebackPolyExp(self, img_gpu, poly_coefficients_gpu, n, sigma):
        assert img_gpu.size == poly_coefficients_gpu[0].size
        assert img_gpu.dtype == np.float32
        assert poly_coefficients_gpu.dtype == np.float32

        invG, G_half = self._FarnebackPrepareGaussian(n, sigma)

        block = (32, 32, 1)
        grid = (int(divup(img_gpu.shape[2], block[0])),
                int(divup(img_gpu.shape[1], block[1])), 1)

        cuda.memcpy_htod(self._invG_gpu, invG)
        cuda.memcpy_htod(self._weights_gpu, G_half)

        self._poly_expansion_kernel(img_gpu,
                                    poly_coefficients_gpu,
                                    np.int32(img_gpu.shape[2]),
                                    np.int32(img_gpu.shape[1]),
                                    np.int32(img_gpu.shape[0]),
                                    np.int32(2 * n + 1),
                                    grid=grid, block=block)

    def _FarnebackUpdateMatrices_gpu(self, R0_gpu, R1_gpu, flow_gpu, M_gpu):

        R1_warped_gpu = gpuarray.empty_like(R1_gpu)

        block = (32, 32, 1)
        grid = (int(divup(flow_gpu.shape[3], block[0])),
                int(divup(flow_gpu.shape[2], block[1])), 1)

        for i in range(_NUM_POLY_COEFFICIENTS - 1):
            utils.ndarray_to_float_tex(
                self._r1_texture, R1_gpu[i])
            self._warp_kernel(
                flow_gpu,
                R1_warped_gpu[i],
                np.int32(flow_gpu.shape[3]),
                np.int32(flow_gpu.shape[2]),
                np.int32(flow_gpu.shape[1]),
                np.float32(1),
                np.float32(1),
                np.float32(1),
                block=block, grid=grid)

        self._update_matrices_kernel(R0_gpu,
                                     R1_warped_gpu,
                                     flow_gpu,
                                     M_gpu,
                                     np.int32(flow_gpu.shape[3]),
                                     np.int32(flow_gpu.shape[2]),
                                     np.int32(flow_gpu.shape[1]),
                                     block=block, grid=grid)

    def _FarnebackUpdateFlow_GaussianBlur_gpu(self, poly_coefficients0, poly_coefficients1, flow_gpu, M, winsize, update_matrices):
        sigma = self.winsize * 0.3

        M_filtered_gpu = gpuarray.GPUArray(M.shape, M.dtype)

        for i in range(M.shape[0]):
            filtering.smooth_cuda_gauss(
                M[i], sigma, winsize, rtn_gpu=M_filtered_gpu[i])

        block = (32, 32, 1)
        grid = (int(divup(flow_gpu.shape[3], block[0])),
                int(divup(flow_gpu.shape[2], block[1])), 1)

        self._solve_equations_kernel(M_filtered_gpu, flow_gpu, np.int32(flow_gpu.shape[3]), np.int32(
            flow_gpu.shape[2]), np.int32(flow_gpu.shape[1]), block=block, grid=grid)

        if update_matrices:
            self._FarnebackUpdateMatrices_gpu(
                poly_coefficients0, poly_coefficients1, flow_gpu, M)

    def _resize(self, src_vol, dst_vol=None, dst_shape=None, scaling=None):
        return utils.resize_gpu(src_vol, dst_vol, dst_shape, scaling)

