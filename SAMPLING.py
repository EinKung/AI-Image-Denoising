from torch.utils.data import Dataset
import PIL.Image as image
import UTILS as utils
import numpy as np
import os
import CONFIGURATION as config

class Sampling_ID(Dataset):
    def __init__(self,dataDir):
        self.filepath=list(map(lambda file:os.path.join(dataDir,file),os.listdir(dataDir)))

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, index):
        orgData=utils.picCut(np.array(image.open(self.filepath[index]).convert('RGB')),config.crop_size_id)        # RGB
        noiData=utils.addGaussianNoise(orgData,1)
        return noiData.astype(np.float32).transpose([2,0,1])/255-0.5,orgData.astype(np.float32).transpose([2,0,1])/255-0.5

class Sampling_ESR(Dataset):
    def __init__(self,dataDir):
        self.filepath=list(map(lambda filename:os.path.join(dataDir,filename),os.listdir(dataDir)))

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, index):
        cropSize=utils.sizeRecurrect(config.crop_size_esr,config.up_scale)
        hrPic=utils.picCut(np.array(image.open(self.filepath[index]).convert('RGB')),cropSize)
        lrPic=utils.resize(hrPic,(cropSize//config.up_scale,cropSize//config.up_scale))
        return lrPic.astype(np.float32).transpose([2,0,1])/255-0.5,hrPic.astype(np.float32).transpose([2,0,1])/255-0.5
