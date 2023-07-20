import os
from tqdm import tqdm
import gzip
import numpy as np
import matplotlib.pyplot as plt
import imageio

folder_path=r"D:\GP\3D ISA\python\positions"
files = os.listdir(folder_path)
print(files)
Positions=[]
first_volume=imageio.imread(r'D:\GP\3D ISA\videos\heart_LV_chr01_04_100_r4.tiff')[0]
print(np.shape(first_volume))

x_range =40.371235
y_range =47.225006
z_range =48.36529

shape=202,237,242

map_values={'x':x_range/202,'y':y_range/237,'z':z_range/242}

map_value=0.2
for index,file in tqdm(enumerate(files)) :
    file_path=os.path.join(folder_path,file)

    file = gzip.GzipFile(file_path , "r")  
    # read the data and append it in the global variable with shape (3 , height * width * depth)         
    position = np.load(file , allow_pickle = True)
    
    Positions.append(position)

def converte_to_xyz(Positions,points):
    #positions shape(time,axis,point)
    #points shape(points,axis)
    
    xyz=[]
    drop_index=[]
    for i,point in enumerate(points):

        index=np.where(np.all(Positions[0]==point,axis=-1))
        if(any(index)):
            xyz.append(Positions[:,index,:].squeeze())
            #print(np.shape(xyz))
        else:
            drop_index.append(i)

    return np.asarray(xyz),drop_index

#true_xyz = np.load(r'D:\GP\3D ISA\python\xyz_test.npz' )['arr_0']
true_xyz = np.load(r'D:\GP\3D ISA\python\heart_LV_chr01_04_100_r4.npz' )['arr_0']


print("shape youssef data " + str(true_xyz.shape))
Positions=np.array(Positions).swapaxes(-1,-2)
print("positions new shape = ", str(Positions.shape))

xyz,drop_list=converte_to_xyz(Positions,true_xyz[0])
print("aboda"+str(xyz.shape))
true_xyz=np.delete(true_xyz,drop_list,axis=1)
true_xyz=true_xyz.swapaxes(0,1)
print(true_xyz.shape,xyz.shape)
data=[]
# for i in range(5):#loop over time 
#     error=((((true_xyz[:,i]-true_xyz[:,i+1])-(xyz[:,i]-xyz[:,i+1]))**2).sum(-1)**(1/2)).mean()
#     data.append(error)
# avg_total_error=(((true_xyz-xyz)**2).sum(-1)**(1/2)).mean()
# print(avg_total_error)

#dist for points in time step=1
#DATA=(((true_xyz[:,1]-xyz[:,1])**2).sum(-1)**(1/2))
#sum of dist for all points at each time step
#DATA=(((true_xyz[:,:]-xyz[:,:])**2).sum(-1).mean(0)**(1/2))
#data= ((((true_xyz[:,0]-true_xyz[:,1])-(xyz[:,0]-xyz[:,1]))**2).sum(-1)**(1/2))
data= ((((true_xyz[:,:20]-xyz[:,:])*0.25)**2).sum(-1)**(1/2)).mean(1) # error for time step 1
data= ((((true_xyz[:,20]-xyz[:,0])*0.25)**2).sum(-1)**(1/2)) # error for time step 1

#data= (((true_xyz-xyz).mean(1)*0.2)**2).sum(-1)**(1/2) # error for all time steps 

#data= (((true_xyz[:,-1]-xyz[:,-1])*0.2)**2).sum(-1)**(1/2) # error for a certain time step
#data= (((((true_xyz[:,0]-true_xyz[:,1])-(xyz[:,0]-xyz[:,1]))*0.2)**2).sum(-1)**(1/2)) #error for flow

#print(DATA.max(),(np.array([202, 237, 242])**2).sum()**(1/2))
print(data.shape)
print(data.mean())
print(data.max())
print(data.min())


#print(np.arange(1,len(DATA)+2,1))
##plt.figure()
#plt.scatter(np.arange(0,len(DATA),1),DATA,s=1)
##plt.scatter(np.arange(0,len(data),1),data,s=1)

##plt.show()


r=3
all=data.shape[0]
print(np.shape(np.where(data<r))[-1]/all)
r=5
print(np.shape(np.where(data<r))[-1]/all)
r=8
print(np.shape(np.where(data<r))[-1]/all)
