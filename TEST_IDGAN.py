import UTILS as utils
import CONFIGURATION as config
import torch
from SAMPLING import Sampling_ID
from torch.utils.data import DataLoader

def test():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=torch.load(config.network_idg_path,map_location=device.type)

    spl=Sampling_ID(config.test_dir)
    dataset=DataLoader(dataset=spl,batch_size=config.batch_size_srgan,shuffle=True)

    ssim_sum=0.
    num=0
    for lr,hr in dataset:
        lr,hr=lr.to(device),hr.to(device)
        fake_hr=net(lr)
        ssim_sum+=utils.qualityRank(fake_hr,hr)
        num+=lr.size()[0]

    print('SSIM FOR IDGAN : {}'.format(ssim_sum/float(num)))

if __name__ == '__main__':
    test()
