from Utils import Optical_FLow

#path to your tiff file, I assume the npz file path would be tiff_file_path.replace("tiff","npz")
tiff_file_path= 'TIFF/heart_LV_chr01_02_100_res7.tiff'

tiff=Optical_FLow(tiff_file_path ,num_of_volumes=20)
tiff.optical_flow()
tiff.Get_Trajectories()
tiff.validate(time_step=20,avg=True,scale=0.25)
