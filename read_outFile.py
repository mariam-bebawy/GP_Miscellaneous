import os
import re
import numpy as np
import core
from core import inference
from core import file_utils
import pathlib 
from pylab import *
from matplotlib import colors
import matplotlib.pyplot as plt 
import sys
# from numba import jit, cuda

import pickle


# Your TensorFlow code here
print("hello")

cwd = os.getcwd()
print(cwd)

target_backend='cuda'                        

output_directory = str("F:/SBME 4/GP/codes/hidpy/hidpy-main")
models_selected = ['D','DA','V','DV','DAV'] 
pixel_size = 0.088
pixel_threshold = 150
dt = 149

prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(output_directory).stem, pixel_size, dt, pixel_threshold)

# Open the file containing the arrays
with open("F:/SBME 4/GP/codes/hidpy/hidpy-main/msd_data.txt") as file:
    # Read the contents of the file
    contents = file.read()

# remove the brackets and split into individual matrices
matrices = contents.replace('[','').split(']')

# remove any empty strings
matrices = list(filter(None, matrices))

# convert the matrices to numpy arrays
arrays = []
for matrix_str in matrices:
    matrix_list = [row.split(',') for row in matrix_str.split(';')]
    matrix_array = np.array(matrix_list, dtype=float)
    arrays.append(matrix_array)

# stack the matrices together
numpy_arrays = np.stack(arrays)

print(numpy_arrays.shape)
print('* Fitting the MSDs models using Bayesian inference')
# warnings.filterwarnings('ignore') # Ignore all the warnings 
"""
Bayes = inference.apply_bayesian_inference(numpy_arrays, dt, models_selected )

# Create the pickle directory 
# pickle_directory = output_directory
# file_utils.create_directory(pickle_directory)

"""
pickle_directory = '%s/pickle' % output_directory

# Save the pickle file 
#with open('%s/%s.pickle' % (pickle_directory, prefix), 'wb') as f:
#    pickle.dump(Bayes, f)

with open('%s/%s.pickle' % (pickle_directory, prefix), 'rb') as f:
  bayes =   pickle.load(f)


mask_matrix = np.zeros((numpy_arrays[0].shape[0], numpy_arrays[0].shape[1]))
#mask_matrix[np.where(mask_nucleoli == 1) ] = 100

# Get the diffusion constant map (D)
diffusion_constant_matrix = bayes['D']
diffusion_constant_matrix[np.where(bayes['model'] == 0)] = np.nan
diffusion_constant_matrix[np.where(bayes['D'] < 1e-10)] = np.nan

# Get the anomalous exponent matrix (A)
anomalous_exponent_matrx = bayes['A']
anomalous_exponent_matrx[np.where(bayes['model'] == 0)] = np.nan
anomalous_exponent_matrx[np.where(bayes['A'] < 1e-10)] = np.nan

# Get the drift velocity matrix (V)
drift_velocity_matrix = bayes['V']
drift_velocity_matrix[np.where(bayes['model'] == 0)] = np.nan
drift_velocity_matrix[np.where(bayes['V']==0)] = np.nan

# Plot the model selection image 
model_selection_image_prefix = '%s_model_selection' % prefix
core.plotting.plot_model_selection_image(
    model_selection_matrix=bayes['model'], mask_matrix=mask_matrix, 
    output_directory=output_directory, frame_prefix=model_selection_image_prefix, 
    font_size=14, title='Model Selection', tick_count=3)

# Plot the diffusion constant matrix
d_map_image_prefix = '%s_diffusion_constant_matrix' % prefix
core.plotting.plot_matrix_map(
    matrix=diffusion_constant_matrix, mask_matrix=mask_matrix, 
    output_directory=output_directory, frame_prefix=d_map_image_prefix, 
    font_size=14, title=r'Diffusion Constant ($\mu$m$^2$/s)', tick_count=3)

# Plot the anomalous matrix
a_map_image_prefix = '%s_anomalous_matrix' % prefix
core.plotting.plot_matrix_map(
    matrix=anomalous_exponent_matrx, mask_matrix=mask_matrix, 
    output_directory=output_directory, frame_prefix=a_map_image_prefix, 
    font_size=14, title='Anomalous Exponent', tick_count=3)

# Plot the drift velocity matrix
v_map_image_prefix = '%s_drift_velocity_matrix' % prefix
core.plotting.plot_matrix_map(
    matrix=drift_velocity_matrix, mask_matrix=mask_matrix, 
    output_directory=output_directory, frame_prefix=v_map_image_prefix, 
    font_size=14, title=r'Drift Velocity ($\mu$m/s)', tick_count=3)

