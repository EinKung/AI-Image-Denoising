from NETWORK import Generator_ESR,Discriminator_ESR
import torch,os
import torch.optim as opt
from torch.utils.data import DataLoader
import CONFIGURATION as config
from SAMPLING import Sampling_ESR
import UTILS as utils

def train(new=False):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Generator=torch.load(config.network_srg_path,map_location=device.type) if new and os.path.exists(config.network_srg_path) else Generator_ESR().to(device)
    Discriminator=torch.load(config.network_srd_path,map_location=device.type) if new and os.path.exists(config.network_srd_path) else Discriminator_ESR().to(device)

    optimizerGen=opt.Adam(Generator.parameters())
    optimizerDis=opt.Adam(Discriminator.parameters())

    spl=Sampling_ESR(config.train_dir)
    dataset=DataLoader(dataset=spl,batch_size=config.batch_size_srgan,shuffle=True,num_workers=4)

    epoch=0
    while True:
        Generator.train()
        Discriminator.train()
        for no,(lrPic,hrPic) in enumerate(dataset):
            lrPic,hrPic=lrPic.to(device),hrPic.to(device)

            ############################## Discriminator ###############################
            fakeHR=Generator(lrPic)
            fakeHR_Prediction_Dis=Discriminator(fakeHR).mean()
            realHR_Prediction_Dis=Discriminator(hrPic).mean()

            real_RelativisticLoss=1-(realHR_Prediction_Dis-fakeHR_Prediction_Dis)
            fake_RelativisticLoss_Dis=fakeHR_Prediction_Dis-realHR_Prediction_Dis
            lossDiscriminator=real_RelativisticLoss+fake_RelativisticLoss_Dis

            optimizerDis.zero_grad()
            lossDiscriminator.backward()
            optimizerDis.step()

            ############################## Generator ###############################
            fakeHR_Prediction_Gen=Discriminator(fakeHR)
            realHR_Prediction_Gen=Discriminator(hrPic)

            fake_RelativisticLoss_Gen=1-(fakeHR_Prediction_Gen-realHR_Prediction_Gen)
            lossGenerator=config.alphaADV_ESR*fake_RelativisticLoss_Gen+config.alphaPIX_ESR*utils.pixelLoss(hrPic,fakeHR)+\
                          config.alphaFEA_ESR*utils.featureLoss(hrPic,fakeHR,device,config.num_vggLayer_srgan)+config.alphaSMO_ESR*utils.smoothLoss_ESR(fakeHR)

            optimizerGen.zero_grad()
            lossGenerator.backward()
            optimizerGen.step()
            print('{}_{}_{}_{}'.format(epoch,no,lossDiscriminator,lossGenerator))

        torch.save(Discriminator,config.network_srd_path)
        torch.save(Generator,config.network_srg_path)
        epoch+=1

if __name__ == '__main__':
    train()
