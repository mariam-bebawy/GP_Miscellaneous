import os, sys
import time
import pathlib, pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
from pylab import *
from pathlib import Path
from zipfile import ZipFile

sys.path.append("./repo_HiDpy")
from repo_HiDpy import *
from repo_HiDpy import core
from repo_HiDpy.core import inference, file_utils, plotting

cwd = os.getcwd()
print(f"current working directory = {cwd}")


target_backend='cuda'                        

outputDIR = str("./code_output")
pickleDIR = '%s/pickle' % outputDIR
models_selected = ['D','DA','V','DV','DAV'] 
pixel_size = 0.088
pixel_threshold = 150
dt = 149

prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(outputDIR).stem, pixel_size, dt, pixel_threshold)

# Open the file containing the arrays
print("reading zip file")
start = time.time()
with ZipFile("./code_misc/msd_data.zip", "r") as zip:
    # Read the contents of the file
    contents = zip.read("msd_data.txt").decode(encoding="utf-8")
end = time.time()
print(f"time to read file = {(end - start):.2f} sec")
print(type(contents))

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
print(f"array shape = {numpy_arrays.shape}")

################################################################
################################################################
################################################################

print("* Fitting the MSDs models using Bayesian inference")

print(f"time to calculate bayesian inference using {os.cpu_count()} cores based on a previous run = 1.55 hr")

with open(Path(outputDIR) / "msd_bayes.pickle", "rb") as f:
    bayes = pickle.load(f)
print(f"pickle file read successfully !")


################################################################
################################################################
################################################################


mask_matrix = np.zeros((numpy_arrays[0].shape[0], numpy_arrays[0].shape[1]))

# Get the diffusion constant map (D)
diffusion_constant_matrix = bayes['D']
diffusion_constant_matrix[np.where(bayes['model'] == 0)] = np.nan
diffusion_constant_matrix[np.where(bayes['D'] < 1e-10)] = np.nan
print(f"got it !")

# Get the anomalous exponent matrix (A)
anomalous_exponent_matrx = bayes['A']
anomalous_exponent_matrx[np.where(bayes['model'] == 0)] = np.nan
anomalous_exponent_matrx[np.where(bayes['A'] < 1e-10)] = np.nan
print(f"got it !")

# Get the drift velocity matrix (V)
drift_velocity_matrix = bayes['V']
drift_velocity_matrix[np.where(bayes['model'] == 0)] = np.nan
drift_velocity_matrix[np.where(bayes['V']==0)] = np.nan
print(f"got it !")

################################################################
################################################################
################################################################

# Plot the model selection image 
model_selection_image_prefix = '%s_model_selection' % prefix
plotting.plot_model_selection_image(
    model_selection_matrix=bayes['model'], mask_matrix=mask_matrix, 
    output_directory=outputDIR, frame_prefix=model_selection_image_prefix, 
    font_size=14, title='Model Selection', tick_count=3)
print(f"plotted !")

# Plot the diffusion constant matrix
d_map_image_prefix = '%s_diffusion_constant_matrix' % prefix
plotting.plot_matrix_map(
    matrix=diffusion_constant_matrix, mask_matrix=mask_matrix, 
    output_directory=outputDIR, frame_prefix=d_map_image_prefix, 
    font_size=14, title=r'Diffusion Constant ($\mu$m$^2$/s)', tick_count=3)
print(f"plotted !")

# Plot the anomalous matrix
a_map_image_prefix = '%s_anomalous_matrix' % prefix
plotting.plot_matrix_map(
    matrix=anomalous_exponent_matrx, mask_matrix=mask_matrix, 
    output_directory=outputDIR, frame_prefix=a_map_image_prefix, 
    font_size=14, title='Anomalous Exponent', tick_count=3)
print(f"plotted !")

# Plot the drift velocity matrix
v_map_image_prefix = '%s_drift_velocity_matrix' % prefix
plotting.plot_matrix_map(
    matrix=drift_velocity_matrix, mask_matrix=mask_matrix, 
    output_directory=outputDIR, frame_prefix=v_map_image_prefix, 
    font_size=14, title=r'Drift Velocity ($\mu$m/s)', tick_count=3)
print(f"plotted !")
