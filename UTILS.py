import cv2,os
import CONFIGURATION as config
import torch,random,datetime
from NETWORK import VGG16
import numpy as np
from skimage.measure import compare_ssim

def picCut(picMat,threahold):
    height,width,channel=picMat.shape
    minSide=min(height,width)
    if minSide<=threahold:
        x,y=random.randint(0,width-minSide+1),random.randint(0,height-minSide+1)
        return resize(picMat[y:y+minSide,x:x+minSide,:],[threahold,threahold])
    else:
        x,y=random.randint(0,width-threahold+1),random.randint(0,height-threahold+1)
    return picMat[y:y+threahold,x:x+threahold,:]

def addSaltNoise(picMat,dop):
    height,width,channel=picMat.shape
    randSalt=np.random.randint(-100,101,[height,width])
    randSalt[randSalt<-config.dop_salt[dop]]=-255
    randSalt[randSalt>config.dop_salt[dop]]=255
    randSalt=(randSalt/255).astype(np.int)*255
    saltCover=np.repeat(randSalt.reshape([height,width,1]),repeats=channel,axis=-1)
    picNoised=picMat+saltCover
    picNoised[picNoised<0]=0
    picNoised[picNoised>255]=255
    return picNoised.astype(np.uint8)

def addGaussianNoise(picMat,dop):
    height,width,channel=picMat.shape
    gaussCover=np.random.normal(0,config.dop_gauss[dop],height*width*channel).reshape([height,width,channel])
    picNoised=picMat+gaussCover
    picNoised[picNoised<0]=0
    picNoised[picNoised>255]=255
    return picNoised.astype(np.uint8)

def addPoissonNoise(picMat,dop):
    height,width,channel=picMat.shape
    poissonCover=np.random.poisson(lam=config.dop_poisson[dop],size=[height,width,channel])
    picNoised=picMat+poissonCover
    picNoised[picNoised>255]=255
    return picNoised.astype(np.uint8)

def resize(picMat,size):
    return cv2.resize(picMat,dsize=(size[0],size[1]),interpolation=config.interpolation)

def pixelLoss(label,prediction):
    return L2Dis(label,prediction)

def featureLoss(label,prediction,device,num_layer):
    vgg=VGG16(num_layer).to(device)
    channel_count=label.size()[1]
    return L2Dis(vgg(label),vgg(prediction))/channel_count

def smoothLoss(picMat):
    base_horizontal=picMat[...,:-1]
    cover_horizontal=picMat[...,1:]
    base_vertical=picMat[...,1:,:]
    cover_vertical=picMat[...,:-1,:]
    return L2Dis(base_horizontal,cover_horizontal)+L2Dis(base_vertical,cover_vertical)

def smoothLoss_ESR(picMat):
    batch_size = picMat.size()[0]
    count_horizontal = sizeCount(picMat[:, :, 1:, :])
    count_vertical = sizeCount(picMat[:, :, :, 1:])
    base_horizontal=picMat[...,:-1]
    cover_horizontal=picMat[...,1:]
    base_vertical=picMat[...,1:,:]
    cover_vertical=picMat[...,:-1,:]
    smooth_horizontal = L2Dis(base_horizontal,cover_horizontal)/count_horizontal
    smooth_vertical = L2Dis(base_vertical,cover_vertical)/count_vertical
    return 2*config.alphaSMO_ESR*(smooth_horizontal+smooth_vertical)/batch_size

def sizeCount(mat):
    return mat.size()[1] * mat.size()[2] * mat.size()[3]

def L2Dis(x1,x2):
    return torch.mean(torch.sqrt((x1-x2)**2))

def thumbnail(filePath):
    picMat=cv2.imread(filePath)
    fileName=filePath.split('\\')[-1]
    height,width,channel=picMat.shape
    maxSide_pic=max(height,width)
    downScale=float(maxSide_pic)/470. if maxSide_pic==height else float(maxSide_pic)/460.
    picMat=resize(picMat,(int(width/downScale),int(height/downScale)))
    thumbnailPath=os.path.join(config.cache_dir,'thumbnail_{}'.format(fileName))
    cv2.imwrite(thumbnailPath,picMat)
    return thumbnailPath

def sizeRecurrect(cropSize,scale):
    return cropSize-(cropSize%scale)

def logMaker(type,messege,details=None):
    with open(config.logFile_path,'a+') as log:
        log.write(str(datetime.datetime.now())+' [{}]: '.format(type)+messege+'\n')
        if details!=None:
            for detail in details:
                log.write('* '+detail+'\n')

def completionCheck():
    flag=1
    mark=[]
    for path in config.checkList:
        result=os.path.exists(path)
        flag*=result
        if not result:
            mark.append(path)

    return bool(flag),mark

def qualityRank(pic01,pic02):
    img1=pic01.cpu().detach().numpy().astype(np.uint8).transpose([0,2,3,1])
    img2=pic02.cpu().detach().numpy().astype(np.uint8).transpose([0,2,3,1])
    ssim=0.
    for im1,im2 in zip(img1,img2):
        ssim+=compare_ssim(im1,im2 ,data_range=255,multichannel=True)
    return ssim
