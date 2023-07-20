import numpy as np
import os
from tqdm import tqdm
import gzip
import pickle as pk

folder_path=r"D:\GP\3D ISA\python\positions"
files = os.listdir(folder_path)
positions=[]
for index,file in tqdm(enumerate(files)) :
    file_path=os.path.join(folder_path,file)
    with gzip.open(file_path, 'rb') as f:
      position = pk.load(f)
    positions.append(position)

    
print(np.shape(positions))
print(np.shape(np.where((positions[0][0][:]-positions[1][0][:])!=0.)))