import os, sys
import time
import pathlib, pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
from pylab import *
from pathlib import Path

sys.path.append("./repo_HiDpy")
from repo_HiDpy import *
from repo_HiDpy import core
from repo_HiDpy.core import inference, file_utils

cwd = os.getcwd()
print(f"current working directory = {cwd}")


target_backend='cuda'                        

outputDIR = str("./code_output")
models_selected = ['D','DA','V','DV','DAV'] 
pixel_size = 0.088
pixel_threshold = 150
dt = 149

prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(outputDIR).stem, pixel_size, dt, pixel_threshold)

# Open the file containing the arrays
start = time.time()
with open("./code_misc/msd_data.txt") as file:
    # Read the contents of the file
    contents = file.read()
end = time.time()
print(f"time to read file = {(end - start):.2f} sec")


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

start = time.time()
bayes = inference.apply_bayesian_inference(numpy_arrays, dt, models_selected)
end = time.time()
print(f"time to calculate bayesian inference using {os.cpu_count()} cores = {(end - start):.2f} sec, = {((end - start)/60/60):.2f} hr")

with open(Path(outputDIR) / "msd_bayes.pickle", "wb") as f:
    pickle.dump(bayes, f)
print(f"pickle file written successfully !")