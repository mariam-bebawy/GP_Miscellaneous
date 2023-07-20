import numpy as np
import os
from tqdm import tqdm
import gzip
import opticalflow3D
import tifffile as tif
import opticalflow3D
import cupy as cp
from cupyx.scipy.interpolate import interpn
import numpy as np
import os
from tqdm import tqdm
import pickle as pk

class Optical_FLow:
  """Farneback3D class used to instantiate the algorithm with its parameters.

  Args:
      path: path to tiff file
      num_of_volumes: slicing tiff file to this num_of_volumes volumes=tiff[:num_of_volumes] .Defaults to 10
      iters (int): number of iterations. Defaults to 5
      num_levels (int): number of pyramid levels. Defaults to 5
      scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
      spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
          applicability. Defaults to 9.
      presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
      filter_type (str): Defines the type of filter used to average the calculated matrices. Defaults to "box"
      filter_size (int): Size of the filter used to average the matrices. Defaults to 21
  """
  def __init__(self, path, num_of_volumes=10,
              iters=5,
              num_levels=5,
              scale=0.8,
              spatial_size=5,
              presmoothing=3,
              filter_type="box",
              filter_size=9,
               ):
    self.path=path
    self.volumes = tif.imread(path)[:num_of_volumes]
    self.number_of_volumes,self.height,self.width,self.depth = np.shape(self.volumes)
    self.first_volume =self.volumes[0]
    print(np.shape(self.volumes))
    self.fields_folderpath = os.path.join(os.path.dirname(__file__),'fields')
    if not os.path.exists(self.fields_folderpath):
      os.makedirs(self.fields_folderpath)
    self.poistions_folderpath=os.path.join(os.path.dirname(__file__),'positions')
    if not os.path.exists(self.poistions_folderpath):
      os.makedirs(self.poistions_folderpath)
    self.optflow = opticalflow3D.Farneback3D(iters,
                                      num_levels,
                                      scale,
                                      spatial_size,
                                      presmoothing,
                                      filter_type,
                                      filter_size,
                                     )


  def optical_flow(self):
    '''
    calculate flowfields and dump them in flowfields folder

    '''
    vol1 = self.volumes[0]
    # for is implemented to run each and every frame in a loop
    for i in tqdm(range(self.number_of_volumes-1)):
        vol2 = self.volumes[i+1] #.astype(np.float32)
        # calculate frame-to-frame flow between vol0 and vol1
        shape = vol1.shape
        
        self.vz, self.vy, self.vx, _  = self.optflow.calculate_flow(vol1, vol2,total_vol=shape)
        #check for nans
        self.vz[np.isnan(self.vz)] = 0.0
        self.vy[np.isnan(self.vy)] = 0.0
        self.vx[np.isnan(self.vx)] = 0.0
        #filter to limit flow values
        self.vz[self.vz>1] = 1.0
        self.vy[self.vy>1] = 1.0
        self.vx[self.vx>1] = 1.0
        self.vz[self.vz<-1] = -1.0
        self.vy[self.vy<-1] = -1.0
        self.vx[self.vx<-1] = -1.0
        #dump flow
        self.displacements = [self.vx, self.vy, self.vz]
        with gzip.open(os.path.join(os.path.join(self.fields_folderpath,str(str(int(i/100))+str(int(i/10))+str(i%10)+".gz"))), "wb") as f:
          pk.dump(np.asarray(self.displacements), f)
    
        vol1 = vol2

  def Get_Trajectories(self):
    '''
      reconstrunc trajectories and save them in position folder
    '''
    #flowfields
    files = sorted(os.listdir(self.fields_folderpath))

    x=cp.linspace(0 , self.height - 1 ,self.height)
    y=cp.linspace(0 , self.width - 1 , self.width)
    z=cp.linspace(0 , self.depth - 1 , self.depth)
    # points where there is a pead in first volume, get trajectories for these points only
    points = np.where(self.first_volume != 0 )
    #dump first position
    with gzip.open(os.path.join( self.poistions_folderpath,"000.gz"), "wb") as f:
        pk.dump(points, f)

    points = cp.array(points).swapaxes(0,1)
    #interpolating flow and saving next position
    for index,file in tqdm(enumerate(files)) :
        file_path=os.path.join(self.fields_folderpath,file)
        with gzip.open(file_path, 'rb') as f:
          field = pk.load(f)
        intrpolated_flow=cp.array([interpn((x,y,z), cp.array(field[0]), points,method='linear',bounds_error=False,fill_value=0.0),
                                interpn((x,y,z), cp.array(field[1]), points,method='linear',bounds_error=False,fill_value=0.0),
                                interpn((x,y,z), cp.array(field[2]), points,method='linear',bounds_error=False,fill_value=0.0)]).swapaxes(0,1)
        print(file ,np.count_nonzero(intrpolated_flow))

        points= points+intrpolated_flow
        with gzip.open(os.path.join( self.poistions_folderpath,str(str(int((index+1)/100))+str(int((index+1)/10))+str((index+1)%10)+".gz")), "wb") as f:
          pk.dump( points.get().swapaxes(0,1), f)

  def validate(self,time_step=1,avg=False,scale=0.25):
    '''
     compare estimated trajectories with simulated data
    '''
    Positions=[]
    for index,file in tqdm(enumerate(os.listdir((self.poistions_folderpath)))) :
        file_path=os.path.join(self.poistions_folderpath,file)
        file = gzip.GzipFile(file_path , "r")  
        # read the data and append it in the global variable with shape (3 , height * width * depth)         
        position = np.load(file , allow_pickle = True)
        Positions.append(position)

    def _converte_to_xyz(Positions,points):
        xyz=[]
        drop_index=[]
        for i,point in enumerate(points):

            index=np.where(np.all(Positions[0]==point,axis=-1))
            if(any(index)):
                xyz.append(Positions[:,index,:].squeeze())
            else:
                drop_index.append(i)

        return np.asarray(xyz),drop_index

    true_xyz = np.load(self.path.replace('tiff','npz') )['arr_0']


    Positions=np.array(Positions).swapaxes(-1,-2)
    xyz,drop_list=_converte_to_xyz(Positions,true_xyz[0])
    true_xyz=np.delete(true_xyz,drop_list,axis=1)
    true_xyz=true_xyz.swapaxes(0,1)

    def _calc_error(true,estimated):
      if avg:
          return ((((true[:,:time_step]-estimated[:,:time_step])*scale)**2).sum(-1)**(1/2)).mean(1) # avg error 
      else :
         return (((true_xyz[:,time_step]-xyz[:,time_step])*scale)**2).sum(-1)**(1/2)
          

    data=_calc_error(true_xyz,xyz)
    print("AVG TRAJECTORY ERROR = ",data.mean())
    print("MAX TRAJECTORY ERROR " ,data.max())
    print("MIN TRAJECTORY ERROR " ,data.min())
    r=3
    all=data.shape[0]
    print("ACC FOR 3nm THRESHOLD = ", np.shape(np.where(data<r))[-1]/all*100)
    r=5
    print("ACC FOR 5nm THRESHOLD = ", np.shape(np.where(data<r))[-1]/all*100)
    r=8
    print("ACC FOR 8nm THRESHOLD = ", np.shape(np.where(data<r))[-1]/all*100)
                
