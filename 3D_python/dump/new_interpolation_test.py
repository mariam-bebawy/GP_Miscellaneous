import cupy as cp
from cupyx.scipy.interpolate import interpn
import numpy as np
import os
from tqdm import tqdm
import gzip
import pickle as pk
import imageio



first_volume=imageio.imread(r'D:\GP\3D ISA\videos\heart_LV_chr01_04_100_r4.tiff')[0]
print(np.shape(first_volume))
height,width,depth=np.shape(first_volume)

poistions_folderpath=r"D:\GP\3D ISA\python\positions"
flow_folderpath=r"D:\GP\3D ISA\python\fields"
files = sorted(os.listdir(flow_folderpath))

x=cp.linspace(0 , height - 1 ,height)
y=cp.linspace(0 , width - 1 , width)
z=cp.linspace(0 , depth - 1 , depth)
points = np.where(first_volume != 0 )

with gzip.open(os.path.join( poistions_folderpath,"000.gz"), "wb") as f:
    pk.dump(points, f)

points = cp.array(points).swapaxes(0,1)

for index,file in tqdm(enumerate(files)) :
    file_path=os.path.join(flow_folderpath,file)
    with gzip.open(file_path, 'rb') as f:
      field = pk.load(f)
    intrpolated_flow=cp.array([interpn((x,y,z), cp.array(field[0]), points,method='linear',bounds_error=False,fill_value=0.0),
                            interpn((x,y,z), cp.array(field[1]), points,method='linear',bounds_error=False,fill_value=0.0),
                            interpn((x,y,z), cp.array(field[2]), points,method='linear',bounds_error=False,fill_value=0.0)]).swapaxes(0,1)
    print(file ,np.count_nonzero(intrpolated_flow))

    points= points+intrpolated_flow
    with gzip.open(os.path.join( poistions_folderpath,str(str(int((index+1)/100))+str(int((index+1)/10))+str((index+1)%10)+".gz")), "wb") as f:
      pk.dump( points.get().swapaxes(0,1), f)
    