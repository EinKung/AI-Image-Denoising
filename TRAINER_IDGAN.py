from NETWORK import Generator_ID,Discriminator_ID
from SAMPLING import Sampling_ID
import torch,os
from torch.utils.data import DataLoader
import CONFIGURATION as config
import torch.optim as opt
import UTILS as utils

def train(new=False):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Generator=torch.load(config.network_idg_path,map_location=device.type) if new and os.path.exists(config.network_idg_path) else Generator_ID().to(device)
    Discriminator=torch.load(config.network_idd_path,map_location=device.type) if new and os.path.exists(config.network_idd_path) else Discriminator_ID().to(device)
    optimizerGen=opt.Adam(Generator.parameters())
    optimizerDis=opt.Adam(Discriminator.parameters())

    spl=Sampling_ID(config.train_dir)
    dataset=DataLoader(dataset=spl,batch_size=config.batch_size_idgan,shuffle=True,num_workers=4)

    epoch=0
    while True:
        Generator.train()
        Discriminator.train()
        for no,(noi,org) in enumerate(dataset):
            noi,org=noi.to(device),org.to(device)

            ################################# Discriminator #################################
            fake=Generator(noi)
            fakePrediction=Discriminator(fake)
            realPrediction=Discriminator(org)

            lossDis=-torch.mean(torch.log(realPrediction)+torch.log(1.-fakePrediction))
            optimizerDis.zero_grad()
            lossDis.backward()
            optimizerDis.step()

            ################################# Generator #################################
            prediction=Discriminator(fake)

            lossGen=config.alphaADV*-torch.mean(torch.log(prediction))+config.alphaPIX*utils.pixelLoss(org,fake)\
                    +config.alphaFEA*utils.featureLoss(org,fake,device,config.num_vggLayer_idgan)+config.alphaSMO*utils.smoothLoss(fake)
            optimizerGen.zero_grad()
            lossGen.backward()
            optimizerGen.step()
            print('{}_{}_{}_{}'.format(epoch,no,lossDis,lossGen))

        torch.save(Discriminator,config.network_idd_path)
        torch.save(Generator,config.network_idg_path)
        epoch+=1


if __name__ == '__main__':
    train()
